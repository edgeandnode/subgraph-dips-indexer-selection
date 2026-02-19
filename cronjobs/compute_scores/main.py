"""
Entry point for the score computation CronJob.

This script:
1. Validates configuration (fail-fast)
2. Checks if scores have already been computed today (idempotency)
3. Fetches raw data — from BigQuery or Redpanda depending on DATA_SOURCE
4. Runs linear regression and computes all metrics
5. Pre-normalizes static metrics
6. Writes results to BigQuery (bigquery mode) or a JSON file (redpanda mode)
"""

import logging
import os
import resource
import sys
import time
from datetime import date, datetime, timedelta

from bq import BigQueryClient, PermissionError
from processing import compute_all_scores, validate_geoip_databases

logging.basicConfig(
    level=logging.INFO,
    format="%(asctime)s - %(name)s - %(levelname)s - %(message)s",
)
logger = logging.getLogger(__name__)

# Configuration from environment
DATA_SOURCE = os.environ.get("DATA_SOURCE", "bigquery")
BQ_PROJECT = os.environ.get("BQ_PROJECT", "graph-mainnet")
BQ_DATASET = os.environ.get("BQ_DATASET", "iisa_data_for_dips")
BQ_LOCATION = os.environ.get("BQ_LOCATION", "US")
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

    if DATA_SOURCE not in ("bigquery", "redpanda"):
        errors.append(f"DATA_SOURCE must be 'bigquery' or 'redpanda', got '{DATA_SOURCE}'")

    if NUM_DAYS < 1:
        errors.append(f"NUM_DAYS must be >= 1, got {NUM_DAYS}")

    if TARGET_ROWS < 1000:
        errors.append(f"TARGET_ROWS must be >= 1000, got {TARGET_ROWS}")

    if DATA_SOURCE == "redpanda" and not os.environ.get("REDPANDA_BOOTSTRAP_SERVERS"):
        errors.append("REDPANDA_BOOTSTRAP_SERVERS is required when DATA_SOURCE=redpanda")

    if errors:
        for error in errors:
            logger.error(f"Configuration error: {error}")
        raise ConfigurationError(f"Found {len(errors)} configuration error(s)")

    logger.info("Configuration validation passed")


def build_provider(data_source: str):
    """Construct the appropriate data provider for the configured data source."""
    if data_source == "redpanda":
        from redpanda import RedpandaProvider

        logger.info("Using RedpandaProvider as data source")
        return RedpandaProvider()

    logger.info("Using BigQueryClient as data source")
    return BigQueryClient(
        project=BQ_PROJECT,
        dataset=BQ_DATASET,
        location=BQ_LOCATION,
    )


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
        f"Configuration: data_source={DATA_SOURCE}, num_days={NUM_DAYS}, "
        f"target_rows={TARGET_ROWS}"
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

    provider = build_provider(DATA_SOURCE)

    # BigQuery-specific: validate permissions before any expensive operations.
    if isinstance(provider, BigQueryClient):
        try:
            provider.validate_permissions()
        except PermissionError as e:
            logger.error(f"Permission validation failed:\n{e}")
            return 1

    # Check idempotency — skip if already computed today.
    if provider.scores_exist_for_today():
        logger.info("Scores already computed for today, skipping")
        return 0

    # BigQuery-specific: maintain the URL cache used for GeoIP resolution.
    if isinstance(provider, BigQueryClient):
        provider.ensure_url_cache_exists()
        provider.update_url_cache()

    end_date = date.today()
    start_date = end_date - timedelta(days=NUM_DAYS)
    start_ts = start_date.strftime("%Y-%m-%dT%H:%M:%SZ")

    logger.info(f"Computing scores for period: {start_date} to {end_date}")

    # BigQuery-specific: validate source data and URL cache exist cheaply.
    if isinstance(provider, BigQueryClient):
        if not provider.source_data_exists(start_date, NUM_DAYS):
            logger.error(f"No source data found for date range {start_date} to {end_date}")
            return 1

        if not provider.url_cache_has_data():
            logger.error("URL cache is empty - GeoIP resolution will fail for all indexers")
            return 1

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
