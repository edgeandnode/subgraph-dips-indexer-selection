"""
IISA HTTP API - FastAPI endpoints for the Indexing Indexer Selection Algorithm.

This module exposes HTTP endpoints for indexer selection that match the API
contract expected by the Rust HTTP client in dipper-iisa.

Endpoints:
- GET /health - Health check, reports if data is loaded
- POST /scores - Push computed indexer scores from the cronjob (bearer-auth)
- GET /scores/status - Report last computed_at for idempotency (bearer-auth)
- POST /sync-status - Push sync-status snapshot from the fetcher (bearer-auth)
- POST /get-score - Return weighted score and components for one indexer
- POST /select-indexers - Select optimal indexers for a deployment
"""

import hmac
import json
import logging
import os
from contextlib import asynccontextmanager
from datetime import datetime, timezone
from functools import lru_cache
from typing import Any, Optional, cast

import pandas as pd
from fastapi import FastAPI, Header, HTTPException
from pydantic import BaseModel
from pydantic_settings import BaseSettings, SettingsConfigDict

from .indexer_selection import EthAddressStr, IndexerSelector, IpfsHashStr
from .score_loader import DataManager, FileScoreLoader
from .sync_status_loader import SyncStatusData

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
    sync_status_file_path: str = "/app/scores/sync_status.json"
    sync_status_staleness_hours: float = 6.0
    # Bearer token required on POST /scores, POST /sync-status, GET /scores/status.
    # When unset, push endpoints accept unauthenticated requests and a WARNING
    # is logged at startup — local development convenience only.
    push_token: Optional[str] = None
    # When true, startup fails hard if push_token is unset. Set in k8s so a
    # misconfigured Secret is caught at rollout rather than leaving production
    # iisa accepting unauthenticated pushes. Default off for compose/local dev.
    require_push_token: bool = False


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
    max_grt_per_30_days: Optional[float] = None  # e.g., 4500.0
    optimistic_dips_fees: Optional[dict[str, float]] = None  # indexer address -> GRT per 30 days


class SelectedIndexer(BaseModel):
    """Indexer entry in the selection response, including pricing info."""

    id: str
    min_grt_per_30_days: Optional[float] = None
    min_grt_per_billion_entities_per_30_days: Optional[float] = None


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
    sync_status_loaded: bool = False


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


class ScoresStatusResponse(BaseModel):
    """
    Response for the GET /scores/status endpoint.

    Used by the cronjob to decide whether today's scores have already been
    computed and pushed — lets the job skip a redundant run.
    """

    computed_at: Optional[str] = None
    rows: int = 0


class ScoresAcceptedResponse(BaseModel):
    """Response returned by POST /scores on success."""

    status: str
    rows: int


class SyncStatusAcceptedResponse(BaseModel):
    """Response returned by POST /sync-status on success."""

    status: str
    indexers: int


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
        self._sync_status: Optional["SyncStatusData"] = None
        self._initialized: bool = False

    def initialize(self, settings: Settings) -> bool:
        """
        Initialize the IISA providers.

        Returns True if initialization succeeded, False if we should fall back
        to random selection mode.
        """
        self.settings = settings

        try:
            logger.info("Initializing IISA providers...")

            provider = FileScoreLoader()
            logger.info("Score source: cache file (iisa-owned RWO PVC)")

            self.data_manager = DataManager(provider)

            logger.info("IISA providers initialized successfully")
            self._initialized = True
            return True

        except Exception as e:
            logger.warning("Failed to initialize IISA providers: %s", e)
            logger.warning("Service will operate in random selection fallback mode")
            self._initialized = False
            return False

    def refresh_data(self) -> bool:
        """
        Load pre-computed indexer scores from the cache file on disk.

        Called on startup to recover the last successful push. Graceful empty
        fallback is already handled by FileScoreLoader on a cache miss.

        Returns True if scores were loaded successfully, False otherwise.
        """
        if not self._initialized or self.data_manager is None:
            logger.warning("Cannot refresh data: DataManager not initialized")
            return False

        try:
            logger.info("Loading pre-computed indexer scores from cache...")
            success = self.data_manager.load_scores()

            if success:
                self._history = self.data_manager.get_data()
                if self._history is not None:
                    logger.info("Scores loaded successfully: %d indexers", len(self._history))
                    return True

            logger.warning("Failed to load scores")
            return False

        except Exception as e:
            logger.error("Failed to load scores: %s", e)
            return False

    def load_scores_from_records(self, records: list[dict[str, Any]]) -> int:
        """
        Accept a pushed scores payload from the cronjob.

        Dry-run-then-commit ordering: parse the DataFrame, run the
        transform against a local copy to prove it produces a usable
        result, THEN write the payload to the cache file, THEN commit
        the transformed result to in-memory state. A transform failure
        raises before any disk I/O, so the cache file is always either
        the previous valid payload or the new one — never a payload
        that a restart would choke on.

        Returns the number of loaded rows. Raises on parse, transform,
        or write failure. An empty records list propagates to
        transform_scores_df which raises ValueError — the push endpoint
        already 422s on empty bodies, so the direct-call path is the
        only caller that hits that branch and we want it to fail loudly
        rather than silently zero out the cache.
        """
        if self.data_manager is None:
            raise RuntimeError("DataManager not initialized")

        scores_path = self.data_manager.scores_file_path
        if scores_path is None:
            raise RuntimeError("DataManager has no file-backed provider; cannot persist push")

        scores_df = pd.DataFrame(records)
        computed_at = _extract_computed_at(scores_df)

        # Dry-run: transform_scores_df is a pure function that raises on
        # failure without touching state. If the payload is schema-invalid
        # (including empty), this raises and the next two lines never execute.
        transformed = self.data_manager.transform_scores_df(scores_df)

        # Transform succeeded — the payload is known to be loadable.
        # Safe to write to disk; a restart will now reload successfully.
        _atomic_write_json(scores_path, records)

        # Commit the already-validated transformed frame to in-memory state.
        self.data_manager.commit_scores(transformed, computed_at)
        self._history = self.data_manager.get_data()
        return len(self._history) if self._history is not None else 0

    def refresh_sync_status(self) -> bool:
        """Load sync status from the cache file. Returns True on success."""
        if self.settings is None:
            return False

        from .sync_status_loader import SyncStatusLoader

        loader = SyncStatusLoader(self.settings.sync_status_file_path)
        data = loader.load(self.settings.sync_status_staleness_hours)
        if data is not None:
            self._sync_status = data
            return True
        return False

    def load_sync_status_from_dict(self, raw: dict[str, Any]) -> int:
        """
        Accept a pushed sync-status payload.

        Parse-first ordering: constructs a SyncStatusData in memory
        before touching disk. Once parsing succeeds, the raw payload
        is written atomically to the configured cache path and the
        in-memory snapshot is assigned. Returns the number of indexers
        that passed the staleness filter.
        """
        if self.settings is None:
            raise RuntimeError("Settings not initialized")

        data = SyncStatusData(raw, self.settings.sync_status_staleness_hours)

        _atomic_write_json(self.settings.sync_status_file_path, raw)

        self._sync_status = data
        return data.total_indexers

    @property
    def sync_status(self):
        """Get the cached SyncStatusData, or None."""
        return getattr(self, "_sync_status", None)

    @property
    def history(self) -> Optional[pd.DataFrame]:
        """Get the cached history DataFrame."""
        return self._history

    @property
    def is_ready(self) -> bool:
        """Check if the service has data loaded and is ready."""
        return self._history is not None and not self._history.empty


def _atomic_write_json(path: str, payload: Any) -> None:
    """Write JSON to `path` atomically via tmp + os.replace.

    The temporary file lives in the same directory so the rename is within
    a single filesystem. Callers must hold no other reference to `path`
    during the replace.
    """
    parent = os.path.dirname(path)
    if parent:
        os.makedirs(parent, exist_ok=True)
    tmp_path = path + ".tmp"
    with open(tmp_path, "w") as f:
        json.dump(payload, f, default=str)
    os.replace(tmp_path, path)


def _extract_computed_at(scores_df: pd.DataFrame) -> Optional[datetime]:
    """Extract the first record's computed_at as a UTC datetime, or None.

    errors="coerce" turns any unparseable value into NaT, so the pd.isna
    check below captures all parse failures without an explicit except.
    """
    if "computed_at" not in scores_df.columns or scores_df.empty:
        return None
    series = pd.to_datetime(scores_df["computed_at"], utc=True, errors="coerce")
    first = series.iloc[0]
    if pd.isna(first):
        return None
    return first.to_pydatetime()


def _require_push_token(authorization: Optional[str]) -> None:
    """Validate the bearer token against IISA_PUSH_TOKEN.

    When IISA_PUSH_TOKEN is unset, authentication is disabled (local dev
    convenience). Token comparison uses hmac.compare_digest to avoid
    timing oracles. The "Bearer" scheme prefix is matched case-insensitively
    per RFC 6750 §2.1, but the token itself keeps its original case.
    """
    expected = get_settings().push_token
    if not expected:
        return
    if not authorization or authorization[:7].lower() != "bearer ":
        raise HTTPException(status_code=401, detail="Missing or malformed Authorization header")
    provided = authorization[7:].strip()
    if not hmac.compare_digest(provided, expected):
        raise HTTPException(status_code=401, detail="Invalid bearer token")


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
    2. Initialize FileScoreLoader and DataManager (cache-backed, RWO PVC)
    3. Recover the last cached scores + sync-status from disk if present
    4. Warn if IISA_PUSH_TOKEN is unset (auth disabled; acceptable in local dev)

    Under the push model the cronjob POSTs new data directly to this service;
    there is no background polling of the scores file. Restarts recover state
    from the cache mount written by previous successful POSTs.
    """
    global _state

    settings = get_settings()

    # Set log level from settings
    logging.getLogger().setLevel(getattr(logging, settings.log_level))
    logger.setLevel(getattr(logging, settings.log_level))

    logger.info("Starting IISA service...")

    if settings.require_push_token and not settings.push_token:
        logger.critical(
            "IISA_REQUIRE_PUSH_TOKEN is true but IISA_PUSH_TOKEN is unset; "
            "refusing to start. Provision the iisa-push-token Secret or "
            "set IISA_REQUIRE_PUSH_TOKEN=false for local development."
        )
        raise RuntimeError("IISA_PUSH_TOKEN is required but unset")

    if not settings.push_token:
        logger.warning(
            "IISA_PUSH_TOKEN is not set; push endpoints will accept unauthenticated "
            "requests. This is acceptable in local development only."
        )

    # Initialize providers
    if not _state.initialize(settings):
        logger.error("IISA initialization failed - cannot start service")
        raise RuntimeError("Failed to initialize IISA providers")

    # Recover last cached scores from the RWO cache mount. Missing/empty cache
    # is acceptable — the service comes up in fallback mode and the next push
    # from the cronjob will populate it.
    logger.info("Attempting to recover cached scores from disk...")
    if _state.refresh_data():
        logger.info("Recovered cached scores on startup")
    else:
        logger.warning(
            "No cached scores found on startup; serving in random-selection "
            "fallback mode until the first POST /scores arrives."
        )

    # Sync status: also optional on startup
    if _state.refresh_sync_status():
        logger.info("Recovered cached sync status on startup")
    else:
        logger.info("No cached sync status on startup (optional)")

    logger.info("IISA service ready")

    yield

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
        sync_status_loaded=_state.sync_status is not None,
    )


@app.post("/scores", response_model=ScoresAcceptedResponse)
def push_scores(
    payload: list[dict[str, Any]],
    authorization: Optional[str] = Header(None),
) -> ScoresAcceptedResponse:
    """
    Accept a pushed scores snapshot from the cronjob.

    Handler order: validate token → validate body → parse DataFrame →
    atomic write to the cache PVC → update in-memory DataFrame → return
    row count. Parsing before touching disk means a malformed payload
    422s cleanly without mutating on-disk state; the cache is only
    updated once the DataFrame is known to be well-formed.

    Declared as sync def (not async) so FastAPI runs the body in a
    threadpool — the _atomic_write_json call does blocking disk I/O
    (~50-200ms on a 10 MiB payload) which would stall the event loop
    if this were async.
    """
    _require_push_token(authorization)

    if not payload:
        raise HTTPException(
            status_code=422,
            detail="scores payload must be a non-empty list of records",
        )

    if not _state._initialized:
        raise HTTPException(
            status_code=503,
            detail="IISA providers not initialized",
        )

    try:
        rows = _state.load_scores_from_records(payload)
    except Exception:
        logger.exception("Failed to accept pushed scores")
        raise HTTPException(status_code=500, detail="Failed to accept scores")

    logger.info("Accepted pushed scores: %d rows", rows)
    return ScoresAcceptedResponse(status="success", rows=rows)


@app.get("/scores/status", response_model=ScoresStatusResponse)
def scores_status(
    authorization: Optional[str] = Header(None),
) -> ScoresStatusResponse:
    """
    Report the computed_at of the currently loaded scores.

    The cronjob GETs this before running to skip redundant recomputation
    when today's scores are already pushed. Sync def for consistency with
    the other push endpoints — body contains no awaitable work.
    """
    _require_push_token(authorization)

    computed_at: Optional[str] = None
    if _state.data_manager is not None and _state.data_manager._scores_computed_at is not None:
        ts = _state.data_manager._scores_computed_at
        if ts.tzinfo is None:
            ts = ts.replace(tzinfo=timezone.utc)
        computed_at = ts.isoformat()

    rows = len(_state.history) if _state.history is not None else 0
    return ScoresStatusResponse(computed_at=computed_at, rows=rows)


@app.post("/sync-status", response_model=SyncStatusAcceptedResponse)
def push_sync_status(
    payload: dict[str, Any],
    authorization: Optional[str] = Header(None),
) -> SyncStatusAcceptedResponse:
    """
    Accept a pushed sync-status snapshot from the sync_status_fetcher.

    Same parse-first ordering as POST /scores. Empty payloads are
    accepted (means "no indexers currently synced"). Sync def so the
    blocking _atomic_write_json runs in FastAPI's threadpool rather
    than on the event loop.
    """
    _require_push_token(authorization)

    if _state.settings is None:
        raise HTTPException(
            status_code=503,
            detail="IISA service not initialized",
        )

    try:
        indexer_count = _state.load_sync_status_from_dict(payload)
    except Exception:
        logger.exception("Failed to accept pushed sync status")
        raise HTTPException(status_code=500, detail="Failed to accept sync status")

    logger.info("Accepted pushed sync status: %d indexers", indexer_count)
    return SyncStatusAcceptedResponse(status="success", indexers=indexer_count)


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
        ("norm_stake_to_fees", "stake_to_fees"),
        ("norm_base_price_per_epoch", "base_price"),
        ("norm_price_per_entity", "price_per_entity"),
    ]

    for col, name in component_keys:
        if col in row.index and pd.notna(row[col]):
            components[name] = float(row[col])

    # Calculate weighted score using IndexerSelector logic
    from .indexer_selection import DEFAULT_WEIGHTS, _calculate_weighted_score, _normalize_metrics

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

    Note on existing_indexers: This tells IndexerSelector which indexers are currently
    assigned, allowing it to decide whether to add, remove, or replace indexers.
    To get a fresh selection ignoring current assignments, pass existing_indexers: [].
    """
    if not _state.is_ready or _state.history is None:
        raise HTTPException(status_code=503, detail="IISA data not loaded")

    if request.num_candidates <= 0:
        return SelectionResponse(deployment_id=request.deployment_id, indexers=[])

    logger.info(
        "select-indexers request: deployment=%s chain=%s num_candidates=%d "
        "existing=%d blocked=%d budget=%s",
        request.deployment_id,
        request.chain_id,
        request.num_candidates,
        len(request.existing_indexers or []),
        len(request.blocklist or []),
        f"{request.max_grt_per_30_days} GRT/30d" if request.max_grt_per_30_days else "none",
    )

    try:
        response = _select_with_processor(request)
        indexer_ids = [i.id for i in response.indexers]
        logger.info(
            "Selected %d indexers for deployment %s: %s",
            len(response.indexers),
            request.deployment_id,
            indexer_ids,
        )
        return response
    except Exception as e:
        logger.exception("Selection failed for deployment %s", request.deployment_id)
        raise HTTPException(status_code=500, detail=f"Selection failed: {e}")


def _extract_chain_price(dips_min_grt_json: str, chain_id: str) -> Optional[float]:
    """Extract the price for a specific chain from the JSON price map."""
    try:
        prices = json.loads(dips_min_grt_json) if isinstance(dips_min_grt_json, str) else {}
        val = prices.get(chain_id)
        return float(val) if val is not None else None
    except (json.JSONDecodeError, TypeError, ValueError):
        return None


def _filter_by_price(
    history: pd.DataFrame,
    chain_id: Optional[str],
    max_grt_per_30_days: Optional[float],
) -> tuple[pd.DataFrame, str]:
    """
    Filter indexers by DIP pricing constraints.

    Excludes indexers that:
    - Have dips_info_available = False
    - Don't support the requested chain_id
    - Don't have pricing configured for the requested chain_id
    - Have a base price exceeding max_grt_per_30_days for the chain

    Returns:
        Tuple of (filtered DataFrame, filter_reason) where filter_reason is empty
        string if no filtering removed all candidates, otherwise describes why
        all candidates were filtered out.
    """
    if chain_id is None:
        return history, ""

    df = history.copy()
    initial_count = len(df)

    # Only filter if we have the DIP info columns
    if "dips_info_available" not in df.columns:
        return df, ""

    # Exclude indexers without DIP info
    df = df[df["dips_info_available"] == True]  # noqa: E712
    logger.debug("price filter: %d/%d indexers have DIP info", len(df), initial_count)

    if df.empty:
        return df, f"all {initial_count} indexers lack DIP info (dips_info_available=False)"

    # Exclude indexers that don't support the chain
    if "dips_supported_networks" in df.columns:

        def supports_chain(networks_json):
            try:
                networks = json.loads(networks_json) if isinstance(networks_json, str) else []
                return chain_id in networks
            except (json.JSONDecodeError, TypeError):
                return False

        pre_filter = len(df)
        df = df[df["dips_supported_networks"].apply(supports_chain)]
        logger.debug(
            "price filter: %d/%d indexers support chain '%s'", len(df), pre_filter, chain_id
        )
        if df.empty:
            return df, f"none of {pre_filter} indexers support chain '{chain_id}'"

    # Exclude indexers that don't have pricing for this chain
    if "dips_min_grt_per_30_days" in df.columns:

        def has_chain_price(prices_json):
            price_str = _extract_chain_price(prices_json, chain_id)
            return price_str is not None

        pre_filter = len(df)
        df = df[df["dips_min_grt_per_30_days"].apply(has_chain_price)]
        if df.empty:
            return (
                df,
                f"none of {pre_filter} indexers have pricing configured for chain '{chain_id}'",
            )

    if df.empty or max_grt_per_30_days is None:
        return df, ""

    # Exclude indexers whose price exceeds the budget
    max_budget = max_grt_per_30_days

    if "dips_min_grt_per_30_days" in df.columns:

        def within_budget(prices_json):
            price_str = _extract_chain_price(prices_json, chain_id)
            if price_str is None:
                return False
            try:
                return float(price_str) <= max_budget
            except (ValueError, TypeError):
                return False

        pre_filter = len(df)
        df = df[df["dips_min_grt_per_30_days"].apply(within_budget)]
        logger.debug(
            "price filter: %d/%d indexers within budget of %s GRT/30d for chain '%s'",
            len(df),
            pre_filter,
            max_grt_per_30_days,
            chain_id,
        )
        if df.empty:
            return (
                df,
                f"all {pre_filter} indexers exceed payment "
                f"ceiling of {max_grt_per_30_days} GRT/30d "
                f"for chain '{chain_id}'",
            )

    return df, ""


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

    if "dips_min_grt_per_billion_entities_per_30_days" in df.columns:
        df["price_per_entity"] = pd.to_numeric(
            df["dips_min_grt_per_billion_entities_per_30_days"], errors="coerce"
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
                min_grt = _extract_chain_price(
                    row.iloc[0].get("dips_min_grt_per_30_days", "{}"), chain_id
                )
            if "dips_min_grt_per_billion_entities_per_30_days" in row.columns:
                val = row.iloc[0].get("dips_min_grt_per_billion_entities_per_30_days")
                try:
                    min_entity = float(val) if val is not None and pd.notna(val) else None
                except (TypeError, ValueError):
                    min_entity = None

        results.append(
            SelectedIndexer(
                id=idx_id,
                min_grt_per_30_days=min_grt,
                min_grt_per_billion_entities_per_30_days=min_entity,
            )
        )
    return results


def _select_with_processor(request: SelectionRequest) -> SelectionResponse:
    """
    Use IndexerSelector for intelligent indexer selection.

    The IndexerSelector uses weighted scoring based on:
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
    filtered_history, filter_reason = _filter_by_price(
        _state.history,
        request.chain_id,
        request.max_grt_per_30_days,
    )

    if filtered_history.empty:
        if filter_reason:
            logger.warning(
                "No indexers available for deployment %s: %s",
                request.deployment_id,
                filter_reason,
            )
        else:
            logger.warning(
                "No indexers available for deployment %s (unknown reason)",
                request.deployment_id,
            )
        return SelectionResponse(deployment_id=request.deployment_id, indexers=[])

    logger.info(
        "deployment=%s proceeding with %d candidates after price filtering (from %d total)",
        request.deployment_id,
        len(filtered_history),
        len(_state.history) if _state.history is not None else 0,
    )

    # Enrich with chain-specific price columns for normalization
    enriched_history = _enrich_with_chain_prices(filtered_history, request.chain_id)

    # Build existing_agreements dict from request
    existing_agreements: dict[str, list[str]] = {}
    if request.existing_indexers:
        existing_agreements[request.deployment_id] = request.existing_indexers

    # Build pending_agreements dict - convert to expected format
    pending_agreements: dict[str, list[str]] = request.pending_agreements or {}

    # Look up which indexers are already synced for this deployment
    synced_indexers: set[str] = set()
    if _state.sync_status is not None:
        synced_indexers = _state.sync_status.synced_indexers_for(request.deployment_id)
        if synced_indexers:
            logger.info(
                "deployment=%s %d synced indexers available",
                request.deployment_id,
                len(synced_indexers),
            )

    processor = IndexerSelector(
        history=enriched_history,
        deployment_id=cast(IpfsHashStr, request.deployment_id),
        existing_agreements=cast(dict[IpfsHashStr, list[EthAddressStr]], existing_agreements),
        pending_agreements=cast(dict[IpfsHashStr, list[EthAddressStr]], pending_agreements),
        declined_indexers=cast(
            dict[IpfsHashStr, list[EthAddressStr]],
            request.declined_indexers or {},
        ),
        indexer_denylist=cast(list[EthAddressStr], request.blocklist or []),
        target_size=request.num_candidates,
        optimistic_dips_fees=request.optimistic_dips_fees,
        price_ceiling=request.max_grt_per_30_days,
        synced_indexers=cast(set[EthAddressStr], synced_indexers),
    )

    # Log selection reasoning for auditability
    if processor.data is not None and not processor.data.empty and processor.current_group:
        scored = processor.data[processor.data["indexer"].isin(processor.current_group)]
        component_cols = [
            ("norm_stake_to_fees", "stake_to_fees"),
            ("norm_base_price_per_epoch", "base_price"),
            ("norm_lat_lin_reg_coefficient", "latency"),
            ("norm_uptime_score", "uptime"),
            ("norm_success_rate", "success_rate"),
            ("norm_price_per_entity", "price_per_entity"),
        ]
        for _, row in scored.iterrows():
            components = {
                label: round(float(row[col]), 3)
                for col, label in component_cols
                if col in row.index and pd.notna(row[col])
            }
            weighted = (
                round(float(row["weighted_score"]), 4)
                if "weighted_score" in row.index and pd.notna(row["weighted_score"])
                else None
            )
            logger.info(
                "selected indexer=%s score=%.4f components=%s deployment=%s",
                row["indexer"],
                weighted if weighted is not None else 0.0,
                components,
                request.deployment_id,
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
    logger.info("Starting IISA service on %s:%d", settings.host, settings.port)
    uvicorn.run(
        "iisa.iisa_http_endpoints:app",
        host=settings.host,
        port=settings.port,
        reload=False,
    )
