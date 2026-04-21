"""Score computation cronjob entry point.

One-shot: compute indexer quality scores, push the results to the iisa
HTTP service, and exit. Scheduling is handled by the Kubernetes CronJob
(`k8s/score-computation-cronjob.yaml`); this process must not linger
after a run — the pod's 50 GiB memory request is billed for its full
lifetime under GKE Autopilot, so an in-process 24h sleep would waste
roughly 22 of every 24 hours of reserved memory.

Exit codes:
  0 — scores were computed and pushed to iisa
  1 — configuration invalid or run failed (scoring error, push error,
      degraded fallback also failed)
  2 — IISA_REQUIRE_PUSH_TOKEN is true but no token is provisioned
"""

import logging
import os
import random
import resource
import sys
import time
from datetime import date, timedelta

from iisa_client import IISAPushError, get_push_token
from processing import compute_all_scores, compute_degraded_scores, validate_geoip_databases
from redpanda import RedpandaProvider

logging.basicConfig(
    level=logging.INFO,
    format="%(asctime)s - %(name)s - %(levelname)s - %(message)s",
)
logger = logging.getLogger(__name__)

NUM_DAYS = int(os.environ.get("NUM_DAYS", "28"))
TARGET_ROWS = int(os.environ.get("TARGET_ROWS", "20000000"))
GRAPH_NETWORK_SUBGRAPH_URL = os.environ.get("GRAPH_NETWORK_SUBGRAPH_URL", "")
IISA_API_URL = os.environ.get("IISA_API_URL", "")
IISA_REQUIRE_PUSH_TOKEN = os.environ.get("IISA_REQUIRE_PUSH_TOKEN", "").lower() == "true"
IISA_PUSH_TOKEN = get_push_token()


class ConfigurationError(Exception):
    """Raised when required configuration is missing or invalid."""


MODE_FULL = "full"
MODE_PARTIAL = "partial"
MODE_DEGRADED = "degraded"
MODE_FAILED = "failed"


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

    if not IISA_API_URL:
        errors.append("IISA_API_URL is required")

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

    if scores_df is not None and not scores_df.empty:
        try:
            provider.write_scores(scores_df)
        except IISAPushError as e:
            # Push failure (auth, validation, or retry exhaustion) must not
            # escape run accounting. Mark the run failed and let the caller
            # exit non-zero so the CronJob's failedJobsHistoryLimit captures it.
            logger.error("Failed to push scores to iisa: %s", e)
            success = False
            mode = MODE_FAILED

    if mode == MODE_PARTIAL:
        logger.warning(
            "Scoring ran without GeoIP — latency scores are neutral (0.5). "
            "Install MaxMind GeoLite2 databases for full scoring."
        )
    elif mode == MODE_DEGRADED:
        logger.warning("Scoring degraded — full pipeline unavailable, pushed real pricing only.")
    elif mode == MODE_FAILED:
        logger.error("Scoring failed")

    logger.info(
        "Scoring complete: mode=%s, indexers=%d, elapsed=%.1fs, peak_memory=%.0fMB",
        mode,
        len(scores_df) if scores_df is not None else 0,
        elapsed,
        get_peak_memory_mb(),
    )

    return success


def main() -> int:
    """Main entry point for the score computation cronjob."""
    logger.info("Score computation cronjob starting")

    if IISA_API_URL:
        logger.info("IISA push target: %s", IISA_API_URL)

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

    success = run_scoring()
    return 0 if success else 1


if __name__ == "__main__":
    sys.exit(main())
