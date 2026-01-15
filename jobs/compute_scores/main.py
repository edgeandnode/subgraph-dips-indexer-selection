"""
Entry point for the score computation CronJob.

This script:
1. Checks if scores have already been computed today (idempotency)
2. Fetches raw data from BigQuery (~20M rows)
3. Runs linear regression and computes all metrics
4. Pre-normalizes static metrics
5. Writes results to the indexer_scores table
"""

import logging
import os
import sys
from datetime import date, datetime, timedelta

from bq import BigQueryClient
from processing import compute_all_scores

logging.basicConfig(
    level=logging.INFO,
    format="%(asctime)s - %(name)s - %(levelname)s - %(message)s",
)
logger = logging.getLogger(__name__)

# Configuration from environment
BQ_PROJECT = os.environ.get("BQ_PROJECT", "graph-mainnet")
BQ_DATASET = os.environ.get("BQ_DATASET", "iisa_data_for_dips")
BQ_LOCATION = os.environ.get("BQ_LOCATION", "US")
IPINFO_AUTH = os.environ.get("IPINFO_AUTH", "")
NUM_DAYS = int(os.environ.get("NUM_DAYS", "28"))
TARGET_ROWS = int(os.environ.get("TARGET_ROWS", "20000000"))


def main() -> int:
    """Main entry point for the score computation job."""
    logger.info("Starting score computation job")
    logger.info(f"Configuration: project={BQ_PROJECT}, dataset={BQ_DATASET}, num_days={NUM_DAYS}")

    # Initialize BigQuery client
    bq_client = BigQueryClient(
        project=BQ_PROJECT,
        dataset=BQ_DATASET,
        location=BQ_LOCATION,
    )

    # Check idempotency - skip if already computed today
    if bq_client.scores_exist_for_today():
        logger.info("Scores already computed for today, skipping")
        return 0

    # Compute timestamps
    end_date = date.today()
    start_date = end_date - timedelta(days=NUM_DAYS)
    start_ts = start_date.strftime("%Y-%m-%dT%H:%M:%SZ")

    logger.info(f"Computing scores for period: {start_date} to {end_date}")

    # Fetch and compute scores
    try:
        scores_df = compute_all_scores(
            bq_client=bq_client,
            start_date=start_date,
            start_ts=start_ts,
            num_days=NUM_DAYS,
            target_rows=TARGET_ROWS,
            ipinfo_auth=IPINFO_AUTH,
        )

        if scores_df.empty:
            logger.warning("No scores computed - empty result")
            return 1

        logger.info(f"Computed scores for {len(scores_df)} indexers")

        # Write scores to BigQuery
        bq_client.write_scores(scores_df)
        logger.info("Successfully wrote scores to BigQuery")

        return 0

    except Exception as e:
        logger.exception(f"Failed to compute scores: {e}")
        return 1


if __name__ == "__main__":
    sys.exit(main())
