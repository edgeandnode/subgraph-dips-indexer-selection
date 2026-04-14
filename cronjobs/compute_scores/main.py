"""Score computation service.

Long-running service that periodically computes indexer quality scores.
When the full scoring pipeline can't run (no GeoIP databases, insufficient
Redpanda data), falls back to degraded mode: equal quality metrics with
real pricing data from indexer /dips/info endpoints.

HTTP endpoints:
  POST /run    -- trigger immediate scoring run
  GET /health  -- healthcheck (503 until first scores written)
  GET /status  -- last run info and next scheduled run
"""

import json
import logging
import os
import random
import resource
import signal
import sys
import threading
import time
from datetime import date, datetime, timedelta, timezone
from http.server import BaseHTTPRequestHandler, HTTPServer

from iisa_client import get_push_token
from processing import compute_all_scores, compute_degraded_scores, validate_geoip_databases
from redpanda import RedpandaProvider

logging.basicConfig(
    level=logging.INFO,
    format="%(asctime)s - %(name)s - %(levelname)s - %(message)s",
)
logger = logging.getLogger(__name__)

# Configuration from environment
NUM_DAYS = int(os.environ.get("NUM_DAYS", "28"))
TARGET_ROWS = int(os.environ.get("TARGET_ROWS", "20000000"))
SCORING_INTERVAL = int(os.environ.get("SCORING_INTERVAL", "86400"))
HTTP_PORT = int(os.environ.get("SCORING_HTTP_PORT", "9090"))
GRAPH_NETWORK_SUBGRAPH_URL = os.environ.get("GRAPH_NETWORK_SUBGRAPH_URL", "")
IISA_API_URL = os.environ.get("IISA_API_URL", "")
IISA_REQUIRE_PUSH_TOKEN = os.environ.get("IISA_REQUIRE_PUSH_TOKEN", "").lower() == "true"
IISA_PUSH_TOKEN = get_push_token()


class ConfigurationError(Exception):
    """Raised when required configuration is missing or invalid."""

    pass


# Scoring modes
MODE_FULL = "full"
MODE_PARTIAL = "partial"
MODE_DEGRADED = "degraded"
MODE_FAILED = "failed"

# Shared state
_status_lock = threading.Lock()
_last_run: dict = {}
_next_run_time: datetime | None = None
_consecutive_partial: int = 0
_consecutive_degraded: int = 0
_consecutive_failed: int = 0
_run_event = threading.Event()
_shutdown = threading.Event()

# After this many consecutive non-full runs, /health returns 503
DEGRADED_THRESHOLD = int(os.environ.get("DEGRADED_ALERT_THRESHOLD", "3"))


def _handle_signal(signum, frame):
    logger.info("Received signal %d, shutting down", signum)
    _shutdown.set()
    _run_event.set()


signal.signal(signal.SIGTERM, _handle_signal)
signal.signal(signal.SIGINT, _handle_signal)


def validate_configuration() -> None:
    """Validate required configuration before starting.

    Raises ConfigurationError if any required config is missing or invalid.
    """
    errors = []

    if NUM_DAYS < 1:
        errors.append(f"NUM_DAYS must be >= 1, got {NUM_DAYS}")

    if TARGET_ROWS < 1000:
        errors.append(f"TARGET_ROWS must be >= 1000, got {TARGET_ROWS}")

    if not os.environ.get("REDPANDA_BOOTSTRAP_SERVERS"):
        errors.append("REDPANDA_BOOTSTRAP_SERVERS is required")

    if errors:
        for error in errors:
            logger.error("Configuration error: %s", error)
        raise ConfigurationError(f"Found {len(errors)} configuration error(s)")

    logger.info("Configuration validation passed")


def get_peak_memory_mb() -> float:
    """Get peak memory usage in MB."""
    usage = resource.getrusage(resource.RUSAGE_SELF).ru_maxrss
    if sys.platform == "darwin":
        return usage / 1024 / 1024
    return usage / 1024


def run_scoring() -> bool:
    """Run one scoring cycle. Returns True on success."""
    global _consecutive_partial, _consecutive_degraded, _consecutive_failed

    pipeline_start = time.time()
    logger.info("Starting score computation")

    # Seed RNGs for deterministic scoring given the same input data.
    # Set SCORING_SEED to replay a previous run's exact sampling.
    seed = int(os.environ.get("SCORING_SEED", date.today().strftime("%Y%m%d")))
    random.seed(seed)
    logger.info("RNG seed: %d", seed)

    geoip_available = validate_geoip_databases()
    if not geoip_available:
        logger.warning("GeoIP databases unavailable, latency scores will be neutral")

    provider = RedpandaProvider()
    scores_df = None
    mode = MODE_FAILED

    # Always attempt compute_all_scores — it handles both full and partial (no GeoIP) modes
    try:
        end_date = date.today()
        start_date = end_date - timedelta(days=NUM_DAYS)
        start_ts = start_date.strftime("%Y-%m-%dT%H:%M:%SZ")

        mode_label = "full" if geoip_available else "partial (no GeoIP)"
        logger.info("Attempting %s pipeline for %s to %s", mode_label, start_date, end_date)
        scores_df = compute_all_scores(
            provider=provider,
            start_date=start_date,
            start_ts=start_ts,
            num_days=NUM_DAYS,
            target_rows=TARGET_ROWS,
            geoip_available=geoip_available,
            seed=seed,
        )
        if scores_df.empty:
            logger.warning("Pipeline returned empty results")
            scores_df = None
        else:
            mode = MODE_FULL if geoip_available else MODE_PARTIAL
    except Exception as e:
        logger.warning("Pipeline failed: %s", e)
        scores_df = None

    # Degraded fallback: equal quality metrics + real pricing (no Redpanda data needed)
    if scores_df is None:
        logger.info("Running degraded scoring (equal quality + real pricing)")
        try:
            scores_df = compute_degraded_scores(GRAPH_NETWORK_SUBGRAPH_URL)
            if scores_df is not None and not scores_df.empty:
                mode = MODE_DEGRADED
            else:
                scores_df = None
        except Exception as e:
            logger.exception("Degraded scoring also failed: %s", e)
            scores_df = None

    elapsed = time.time() - pipeline_start
    success = scores_df is not None and not scores_df.empty

    if success:
        provider.write_scores(scores_df)

    # Track consecutive non-full runs (under lock — health endpoint reads these)
    with _status_lock:
        if mode == MODE_FULL:
            _consecutive_partial = 0
            _consecutive_degraded = 0
            _consecutive_failed = 0
        elif mode == MODE_PARTIAL:
            _consecutive_partial += 1
            _consecutive_degraded = 0
            _consecutive_failed = 0
        elif mode == MODE_DEGRADED:
            _consecutive_partial = 0
            _consecutive_degraded += 1
            _consecutive_failed = 0
        else:
            _consecutive_partial = 0
            _consecutive_degraded = 0
            _consecutive_failed += 1

        _last_run.update(
            {
                "time": datetime.now(timezone.utc).isoformat(),
                "success": success,
                "indexers": len(scores_df) if success else 0,
                "elapsed_seconds": round(elapsed, 1),
                "mode": mode,
                "consecutive_partial": _consecutive_partial,
                "consecutive_degraded": _consecutive_degraded,
                "consecutive_failed": _consecutive_failed,
            }
        )

    # Log outside the lock
    if mode == MODE_PARTIAL:
        logger.warning(
            "Scoring ran without GeoIP (%d consecutive partial run(s)). "
            "Latency scores are neutral (0.5). "
            "Install MaxMind GeoLite2 databases for full scoring.",
            _consecutive_partial,
        )
    elif mode == MODE_DEGRADED:
        if _consecutive_degraded >= DEGRADED_THRESHOLD:
            logger.error(
                "Scoring has been degraded for %d consecutive runs. "
                "Full pipeline is not functioning — "
                "investigate GeoIP databases and Redpanda data availability.",
                _consecutive_degraded,
            )
        else:
            logger.warning(
                "Scoring degraded (%d/%d before alert)",
                _consecutive_degraded,
                DEGRADED_THRESHOLD,
            )
    elif mode == MODE_FAILED:
        logger.error("Scoring failed completely for %d consecutive runs", _consecutive_failed)

    logger.info(
        "Scoring complete: mode=%s, indexers=%d, elapsed=%.1fs, peak_memory=%.0fMB",
        mode,
        len(scores_df) if success else 0,
        elapsed,
        get_peak_memory_mb(),
    )

    return success


class RequestHandler(BaseHTTPRequestHandler):
    def do_GET(self):
        if self.path == "/health":
            with _status_lock:
                first_run_done = bool(_last_run)
                degraded = _consecutive_degraded
                failed = _consecutive_failed
            if not first_run_done:
                self._json_response(503, {"status": "waiting_for_first_run"})
                return
            if failed > 0:
                self._json_response(
                    503,
                    {
                        "status": "failing",
                        "consecutive_failed": failed,
                    },
                )
            elif degraded >= DEGRADED_THRESHOLD:
                self._json_response(
                    200,
                    {
                        "status": "degraded",
                        "consecutive_degraded": degraded,
                        "message": "Full pipeline not functioning, serving degraded scores",
                    },
                )
            else:
                self._json_response(200, {"status": "ok"})
        elif self.path == "/status":
            with _status_lock:
                status = {
                    "last_run": dict(_last_run),
                    "next_run_time": _next_run_time.isoformat() if _next_run_time else None,
                    "scoring_interval_seconds": SCORING_INTERVAL,
                    "consecutive_partial": _consecutive_partial,
                    "consecutive_degraded": _consecutive_degraded,
                    "consecutive_failed": _consecutive_failed,
                    "degraded_threshold": DEGRADED_THRESHOLD,
                }
            self._json_response(200, status)
        else:
            self.send_response(404)
            self.end_headers()

    def do_POST(self):
        if self.path == "/run":
            _run_event.set()
            self._json_response(202, {"status": "run_triggered"})
        else:
            self.send_response(404)
            self.end_headers()

    def _json_response(self, code: int, body: dict):
        self.send_response(code)
        self.send_header("Content-Type", "application/json")
        self.end_headers()
        self.wfile.write(json.dumps(body).encode())

    def log_message(self, format, *args):
        pass


def main() -> int:
    """Main entry point for the score computation service."""
    global _next_run_time

    logger.info(
        "Score computation service starting (interval=%ds, http_port=%d)",
        SCORING_INTERVAL,
        HTTP_PORT,
    )

    if IISA_API_URL:
        logger.info("IISA push target: %s", IISA_API_URL)
    else:
        logger.error(
            "IISA_API_URL is not set -- scores cannot be pushed to iisa. "
            "Set IISA_API_URL or the scoring run will fail at write time."
        )

    if IISA_REQUIRE_PUSH_TOKEN and not IISA_PUSH_TOKEN:
        logger.critical(
            "IISA_REQUIRE_PUSH_TOKEN is true but IISA_PUSH_TOKEN is unset; "
            "refusing to run. Provision the iisa-push-token Secret or "
            "set IISA_REQUIRE_PUSH_TOKEN=false for local development."
        )
        return 2

    try:
        validate_configuration()
    except ConfigurationError:
        return 1

    # Start HTTP server in background thread
    server = HTTPServer(("0.0.0.0", HTTP_PORT), RequestHandler)
    server_thread = threading.Thread(target=server.serve_forever, daemon=True)
    server_thread.start()
    logger.info("HTTP server listening on port %d", HTTP_PORT)

    # Initial scoring run
    run_scoring()

    # Scheduling loop — runs until shutdown signal
    while not _shutdown.is_set():
        _next_run_time = datetime.now(timezone.utc) + timedelta(seconds=SCORING_INTERVAL)
        triggered = _run_event.wait(timeout=SCORING_INTERVAL)

        if _shutdown.is_set():
            break

        if triggered:
            _run_event.clear()
            logger.info("Manual scoring run triggered via HTTP")

        run_scoring()

    server.shutdown()
    logger.info("Score computation service stopped")
    return 0


if __name__ == "__main__":
    sys.exit(main())
