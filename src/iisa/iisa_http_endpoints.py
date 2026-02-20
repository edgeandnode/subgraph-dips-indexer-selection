"""
IISA HTTP API - FastAPI endpoints for the Indexing Indexer Selection Algorithm.

This module exposes HTTP endpoints for indexer selection that match the API
contract expected by the Rust HTTP client in dipper-iisa.

Endpoints:
- GET /health - Health check, reports if data is loaded
- POST /refresh - Reload scores from the scores file
- POST /select-indexers - Select optimal indexers for a deployment
"""

import json
import logging
from contextlib import asynccontextmanager
from functools import lru_cache
from typing import Optional

import pandas as pd
from fastapi import FastAPI, HTTPException
from pydantic import BaseModel
from pydantic_settings import BaseSettings, SettingsConfigDict

from .indexer_selection import DataProcessor
from .score_loader import DataManager, FileScoreLoader

__all__ = ["app", "Settings", "get_settings"]


# =============================================================================
# Configuration
# =============================================================================


class Settings(BaseSettings):
    """
    Service configuration loaded from environment variables.

    All settings are prefixed with IISA_ in environment variables.
    For example, IISA_HOST sets the host field.
    """

    model_config = SettingsConfigDict(
        env_prefix="IISA_",
        env_file=".env",
        env_file_encoding="utf-8",
        extra="ignore",
    )

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


class SelectionRequest(BaseModel):
    """
    Request body for indexer selection endpoint.

    Matches the SelectionRequest struct in the Rust HTTP client.
    """

    deployment_id: str
    existing_indexers: Optional[list[str]] = None
    pending_agreements: Optional[dict[str, list[str]]] = None
    num_candidates: int  # Target group size (required)
    blocklist: Optional[list[str]] = None
    declined_indexers: Optional[dict[str, list[str]]] = None
    chain_id: Optional[str] = None  # e.g., "arbitrum-one"
    max_grt_per_30_days: Optional[str] = None  # e.g., "4500"


class SelectedIndexer(BaseModel):
    """Indexer entry in the selection response, including pricing info."""

    id: str
    min_grt_per_30_days: Optional[str] = None
    min_grt_per_million_entities_per_30_days: Optional[str] = None


class SelectionResponse(BaseModel):
    """
    Response for the /select-indexers endpoint.

    Returns the optimal set of indexers that SHOULD be assigned to the deployment.
    This is idempotent - replaying the same request yields the same response.
    The caller diffs against its actual current state to determine adds/cancels.
    """

    deployment_id: str
    indexers: list[SelectedIndexer]


class HealthResponse(BaseModel):
    """
    Response for the /health endpoint.
    """

    status: str
    data_loaded: bool


class ScoreRequest(BaseModel):
    """
    Request body for the /get-score endpoint.
    """

    indexer_id: str


class ScoreResponse(BaseModel):
    """
    Response for the /get-score endpoint.

    Returns the weighted score and component scores for an indexer.
    """

    indexer_id: str
    weighted_score: Optional[float] = None
    components: Optional[dict[str, float]] = None
    found: bool


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

            provider = FileScoreLoader()
            logger.info("Score source: file (shared PVC)")

            self.data_manager = DataManager(provider)

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
        Load pre-computed indexer scores from the scores file.

        This method loads scores from the scores JSON file (populated by CronJob)
        instead of computing them in-container. This is much faster and uses less memory.

        Returns True if scores were loaded successfully, False otherwise.
        """
        if not self._initialized or self.data_manager is None:
            logger.warning("Cannot refresh data: DataManager not initialized")
            return False

        try:
            logger.info("Loading pre-computed indexer scores...")
            success = self.data_manager.load_scores()

            if success:
                self._history = self.data_manager.get_data()
                if self._history is not None:
                    logger.info(f"Scores loaded successfully: {len(self._history)} indexers")
                    return True

            logger.warning("Failed to load scores")
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
    2. Initialize FileScoreLoader and DataManager
    3. Load pre-computed scores from the scores file

    If any step fails, the service continues in fallback mode with random selection.
    """
    global _state

    settings = get_settings()

    # Set log level from settings
    logging.getLogger().setLevel(getattr(logging, settings.log_level))
    logger.setLevel(getattr(logging, settings.log_level))

    logger.info("Starting IISA service...")

    # Initialize providers and load data
    if not _state.initialize(settings):
        logger.error("IISA initialization failed - cannot start service")
        raise RuntimeError("Failed to initialize IISA providers")

    # Load data on startup - service won't accept requests until ready
    logger.info("Loading indexer scores...")
    if not _state.refresh_data():
        logger.error("Failed to load indexer scores - cannot start service")
        raise RuntimeError("Failed to load indexer scores")

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
    Trigger a data refresh from the scores file.

    This endpoint reloads scores from the shared PVC.
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
        raise HTTPException(status_code=500, detail="Failed to refresh data")


@app.post("/get-score", response_model=ScoreResponse)
async def get_score(request: ScoreRequest) -> ScoreResponse:
    """
    Get the weighted score and component scores for an indexer.

    Returns the indexer's current weighted score along with the individual
    component scores that contribute to it. Useful for debugging selection
    decisions and monitoring indexer performance.
    """
    if not _state.is_ready or _state.history is None:
        raise HTTPException(status_code=503, detail="IISA data not loaded")

    # Normalize to lowercase for case-insensitive matching
    indexer_id = request.indexer_id.lower()
    indexer_data = _state.history[_state.history["indexer"] == indexer_id]

    if indexer_data.empty:
        return ScoreResponse(
            indexer_id=request.indexer_id,
            found=False,
        )

    row = indexer_data.iloc[0]

    # Extract component scores (norm_ prefixed columns)
    components = {}
    component_keys = [
        ("norm_lat_lin_reg_coefficient", "latency"),
        ("norm_uptime_score", "uptime"),
        ("norm_success_rate", "success_rate"),
        ("norm_stake_to_fees_iqr_deviation", "stake_to_fees"),
        ("norm_base_price_per_epoch", "base_price"),
        ("norm_price_per_entity", "price_per_entity"),
    ]

    for col, name in component_keys:
        if col in row.index and pd.notna(row[col]):
            components[name] = float(row[col])

    # Calculate weighted score using DataProcessor logic
    from .indexer_selection import _normalize_metrics, _calculate_weighted_score, DEFAULT_WEIGHTS

    # Normalize and calculate score for this single indexer
    normalized = _normalize_metrics(indexer_data.copy())
    if not normalized.empty:
        weighted_score = float(_calculate_weighted_score(normalized.iloc[0], DEFAULT_WEIGHTS))
    else:
        weighted_score = None

    return ScoreResponse(
        indexer_id=request.indexer_id,
        weighted_score=weighted_score,
        components=components,
        found=True,
    )


@app.post("/select-indexers", response_model=SelectionResponse)
async def select_indexers(request: SelectionRequest) -> SelectionResponse:
    """
    Select optimal indexers for a deployment using weighted scoring algorithm.

    Returns the target state - the set of indexers that SHOULD be assigned.
    This is idempotent: replaying the same request yields the same response.
    The caller diffs against its actual current state to determine adds/cancels.

    The num_candidates field specifies the target group size - how many indexers
    should be assigned to this deployment. IISA selects the top N indexers by
    weighted aggregate score, preferring groups with >1 unique org and >1 unique
    location when N > 1. These decentralization constraints are best-effort.

    Note on existing_indexers: This tells DataProcessor which indexers are currently
    assigned, allowing it to decide whether to add, remove, or replace indexers.
    To get a fresh selection ignoring current assignments, pass existing_indexers: [].
    """
    if not _state.is_ready or _state.history is None:
        raise HTTPException(status_code=503, detail="IISA data not loaded")

    if request.num_candidates <= 0:
        return SelectionResponse(deployment_id=request.deployment_id, indexers=[])

    try:
        response = _select_with_processor(request)
        indexer_ids = [i.id for i in response.indexers]
        logger.info(
            f"Selected {len(response.indexers)} indexers for deployment "
            f"{request.deployment_id}: {indexer_ids}"
        )
        return response
    except Exception as e:
        logger.exception(f"Selection failed for deployment {request.deployment_id}")
        raise HTTPException(status_code=500, detail=f"Selection failed: {e}")


def _extract_chain_price(dips_min_grt_json: str, chain_id: str) -> Optional[str]:
    """Extract the price for a specific chain from the JSON price map."""
    try:
        prices = json.loads(dips_min_grt_json) if isinstance(dips_min_grt_json, str) else {}
        return prices.get(chain_id)
    except (json.JSONDecodeError, TypeError):
        return None


def _filter_by_price(
    history: pd.DataFrame,
    chain_id: Optional[str],
    max_grt_per_30_days: Optional[str],
) -> pd.DataFrame:
    """
    Filter indexers by DIP pricing constraints.

    Excludes indexers that:
    - Have dips_info_available = False
    - Don't support the requested chain_id
    - Have a base price exceeding max_grt_per_30_days for the chain
    """
    if chain_id is None:
        return history

    df = history.copy()

    # Only filter if we have the DIP info columns
    if "dips_info_available" not in df.columns:
        return df

    # Exclude indexers without DIP info
    df = df[df["dips_info_available"] == True]  # noqa: E712

    if df.empty:
        return df

    # Exclude indexers that don't support the chain
    if "dips_supported_networks" in df.columns:
        def supports_chain(networks_json):
            try:
                networks = json.loads(networks_json) if isinstance(networks_json, str) else []
                return chain_id in networks
            except (json.JSONDecodeError, TypeError):
                return False

        df = df[df["dips_supported_networks"].apply(supports_chain)]

    if df.empty or max_grt_per_30_days is None:
        return df

    # Exclude indexers whose price exceeds the budget
    max_budget = float(max_grt_per_30_days)

    if "dips_min_grt_per_30_days" in df.columns:
        def within_budget(prices_json):
            price_str = _extract_chain_price(prices_json, chain_id)
            if price_str is None:
                return False
            try:
                return float(price_str) <= max_budget
            except (ValueError, TypeError):
                return False

        df = df[df["dips_min_grt_per_30_days"].apply(within_budget)]

    return df


def _enrich_with_chain_prices(
    history: pd.DataFrame,
    chain_id: Optional[str],
) -> pd.DataFrame:
    """
    Add base_price_per_epoch and price_per_entity columns for scoring.

    Extracts the chain-specific price from the JSON fields.
    """
    df = history.copy()

    if chain_id is None or "dips_min_grt_per_30_days" not in df.columns:
        df["base_price_per_epoch"] = 0.0
        df["price_per_entity"] = 0.0
        return df

    def extract_price(prices_json):
        price_str = _extract_chain_price(prices_json, chain_id)
        try:
            return float(price_str) if price_str is not None else 0.0
        except (ValueError, TypeError):
            return 0.0

    df["base_price_per_epoch"] = df["dips_min_grt_per_30_days"].apply(extract_price)

    if "dips_min_grt_per_million_entities_per_30_days" in df.columns:
        df["price_per_entity"] = pd.to_numeric(
            df["dips_min_grt_per_million_entities_per_30_days"], errors="coerce"
        ).fillna(0.0)
    else:
        df["price_per_entity"] = 0.0

    return df


def _build_selected_indexers(
    indexer_ids: list[str],
    history: pd.DataFrame,
    chain_id: Optional[str],
) -> list[SelectedIndexer]:
    """Build SelectedIndexer entries with pricing info."""
    results = []
    for idx_id in indexer_ids:
        row = history[history["indexer"] == idx_id]
        min_grt = None
        min_entity = None

        if not row.empty and chain_id is not None:
            if "dips_min_grt_per_30_days" in row.columns:
                min_grt = _extract_chain_price(row.iloc[0].get("dips_min_grt_per_30_days", "{}"), chain_id)
            if "dips_min_grt_per_million_entities_per_30_days" in row.columns:
                val = row.iloc[0].get("dips_min_grt_per_million_entities_per_30_days")
                min_entity = str(val) if val is not None and pd.notna(val) else None

        results.append(SelectedIndexer(
            id=idx_id,
            min_grt_per_30_days=min_grt,
            min_grt_per_million_entities_per_30_days=min_entity,
        ))
    return results


def _select_with_processor(request: SelectionRequest) -> SelectionResponse:
    """
    Use DataProcessor for intelligent indexer selection.

    The DataProcessor uses weighted scoring based on:
    - Stake to fees ratio (economic security)
    - Base price per epoch (cheaper is better)
    - Latency linear regression coefficient
    - Uptime score
    - Success rate
    - Price per entity (cheaper is better)

    Returns a SelectionResponse with the optimal set of indexers for the deployment.
    """
    if _state.history is None:
        return SelectionResponse(deployment_id=request.deployment_id, indexers=[])

    # Filter by price constraints before scoring
    filtered_history = _filter_by_price(
        _state.history,
        request.chain_id,
        request.max_grt_per_30_days,
    )

    if filtered_history.empty:
        return SelectionResponse(deployment_id=request.deployment_id, indexers=[])

    # Enrich with chain-specific price columns for normalization
    enriched_history = _enrich_with_chain_prices(filtered_history, request.chain_id)

    # Build existing_agreements dict from request
    existing_agreements: dict[str, list[str]] = {}
    if request.existing_indexers:
        existing_agreements[request.deployment_id] = request.existing_indexers

    # Build pending_agreements dict - convert to expected format
    pending_agreements: dict[str, list[str]] = request.pending_agreements or {}

    # Create DataProcessor instance with target_size from num_candidates
    processor = DataProcessor(
        history=enriched_history,
        deployment_id=request.deployment_id,
        existing_agreements=existing_agreements,
        pending_agreements=pending_agreements,
        declined_indexers=request.declined_indexers or {},
        indexer_denylist=request.blocklist or [],
        target_size=request.num_candidates,
    )

    # Build response with pricing info
    selected = _build_selected_indexers(
        list(processor.current_group),
        enriched_history,
        request.chain_id,
    )

    return SelectionResponse(
        deployment_id=request.deployment_id,
        indexers=selected,
    )


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
