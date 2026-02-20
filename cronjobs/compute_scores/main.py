"""
Entry point for the score computation CronJob.

This script:
1. Validates configuration (fail-fast)
2. Checks if scores have already been computed today (idempotency)
3. Fetches raw data from Redpanda
4. Runs linear regression and computes all metrics
5. Pre-normalizes static metrics
6. Writes results to a JSON file on the shared PVC
"""

import logging
import os
import resource
import sys
import time
from datetime import date, timedelta

from processing import compute_all_scores, validate_geoip_databases
from redpanda import RedpandaProvider

logging.basicConfig(
    level=logging.INFO,
    format="%(asctime)s - %(name)s - %(levelname)s - %(message)s",
)
logger = logging.getLogger(__name__)

# Configuration from environment
NUM_DAYS = int(os.environ.get("NUM_DAYS", "28"))
TARGET_ROWS = int(os.environ.get("TARGET_ROWS", "20000000"))


class ConfigurationError(Exception):
    """Raised when required configuration is missing or invalid."""

    pass


def validate_configuration() -> None:
    """Validate all required configuration before starting expensive operations.

    Raises ConfigurationError if any required config is missing or invalid.
    This implements fail-fast principle - better to fail in 1 second than 30 minutes.
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
            logger.error(f"Configuration error: {error}")
        raise ConfigurationError(f"Found {len(errors)} configuration error(s)")

    logger.info("Configuration validation passed")


def get_peak_memory_mb() -> float:
    """Get peak memory usage in MB."""
    # ru_maxrss is in bytes on Linux, KB on macOS
    usage = resource.getrusage(resource.RUSAGE_SELF).ru_maxrss
    if sys.platform == "darwin":
        return usage / 1024 / 1024  # macOS: bytes -> MB
    return usage / 1024  # Linux: KB -> MB


def main() -> int:
    """Main entry point for the score computation job."""
    pipeline_start = time.time()
    logger.info("Starting score computation job")
    logger.info(
        f"Configuration: num_days={NUM_DAYS}, target_rows={TARGET_ROWS}"
    )

    try:
        validate_configuration()
    except ConfigurationError:
        return 1

    try:
        validate_geoip_databases()
    except FileNotFoundError as e:
        logger.error(f"GeoIP database validation failed: {e}")
        return 1

    provider = RedpandaProvider()

    # Check idempotency — skip if already computed today.
    if provider.scores_exist_for_today():
        logger.info("Scores already computed for today, skipping")
        return 0

    end_date = date.today()
    start_date = end_date - timedelta(days=NUM_DAYS)
    start_ts = start_date.strftime("%Y-%m-%dT%H:%M:%SZ")

    logger.info(f"Computing scores for period: {start_date} to {end_date}")

    try:
        scores_df = compute_all_scores(
            provider=provider,
            start_date=start_date,
            start_ts=start_ts,
            num_days=NUM_DAYS,
            target_rows=TARGET_ROWS,
        )

        if scores_df.empty:
            logger.warning("No scores computed - empty result")
            return 1

        logger.info(f"Computed scores for {len(scores_df)} indexers")

        provider.write_scores(scores_df)

        elapsed = time.time() - pipeline_start
        logger.info(
            f"Pipeline completed in {elapsed:.1f}s ({elapsed/60:.1f}m), "
            f"peak memory: {get_peak_memory_mb():.0f} MB"
        )
        return 0

    except Exception as e:
        elapsed = time.time() - pipeline_start
        logger.exception(f"Failed to compute scores after {elapsed:.1f}s: {e}")
        logger.info(f"Peak memory at failure: {get_peak_memory_mb():.0f} MB")
        return 1


if __name__ == "__main__":
    sys.exit(main())
