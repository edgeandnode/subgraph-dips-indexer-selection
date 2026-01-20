import logging
from datetime import datetime, timedelta, timezone
from typing import Optional

import pandas as pd

from ..bq import BigQueryProvider

__all__ = ["DataManager"]

# Staleness thresholds
STALE_SCORES_WARNING_HOURS = 48
STALE_SCORES_CRITICAL_HOURS = 168  # 7 days

logger = logging.getLogger(__name__)


class DataManager:
    """
    Loads pre-computed indexer scores from BigQuery.

    Scores are computed daily by a CronJob and include latency regression
    coefficients, uptime, success rate, and economic security metrics.
    """

    def __init__(self, bigquery: BigQueryProvider) -> None:
        self._bq = bigquery
        self._data: Optional[pd.DataFrame] = None
        self._scores_computed_at: Optional[datetime] = None

    def load_scores(self) -> bool:
        """
        Load pre-computed indexer scores from BigQuery.

        :return: True if scores were loaded successfully, False otherwise.
        """
        logger.info("Loading pre-computed indexer scores from BigQuery")

        scores_df, computed_at = self._bq.fetch_indexer_scores()

        if scores_df.empty:
            logger.warning("No pre-computed scores available in indexer_scores table")
            self._data = None
            return False

        self._scores_computed_at = computed_at
        self._check_scores_staleness(computed_at)
        self._data = self._transform_scores_to_perf_history(scores_df)

        logger.info(f"Loaded scores for {len(self._data)} indexers")
        return True

    def _transform_scores_to_perf_history(
        self, scores_df: pd.DataFrame
    ) -> pd.DataFrame:
        """
        Transform indexer_scores table format to DataProcessor-compatible format.

        Column mapping:
        - lat_coefficient_upper_bound -> "Latency Coefficient + Error Confidence Interval"
        - uptime_score (0-1) -> "% up_x" (0-100)
        - success_rate -> "average_status"
        - dst_lat, dst_lon -> "destination_loc"
        - norm_stake_to_fees -> "norm_stake_to_fees_iqr_deviation"
        """
        df = scores_df.copy()

        # TODO: Refactor DataProcessor to use CronJob column names directly
        df = df.rename(columns={
            "lat_coefficient_upper_bound": "Latency Coefficient + Error Confidence Interval",
            "success_rate": "average_status",
            "norm_stake_to_fees": "norm_stake_to_fees_iqr_deviation",
            "lat_normalized_score": "norm_lat_lin_reg_coefficient",
        })

        if "uptime_score" in df.columns:
            df["% up_x"] = df["uptime_score"] * 100

        if "dst_lat" in df.columns and "dst_lon" in df.columns:
            df["destination_loc"] = (
                df["dst_lat"].fillna(0).astype(str) + "," +
                df["dst_lon"].fillna(0).astype(str)
            )

        if "existing_dips_agreements" not in df.columns:
            df["existing_dips_agreements"] = 0

        return df

    def _check_scores_staleness(self, computed_at: Optional[datetime]) -> None:
        """Log warnings if pre-computed scores are stale."""
        if computed_at is None:
            logger.warning("Scores have no computation timestamp")
            return

        now = datetime.now(timezone.utc)
        if computed_at.tzinfo is None:
            computed_at = computed_at.replace(tzinfo=timezone.utc)

        age_hours = (now - computed_at).total_seconds() / 3600

        if age_hours > STALE_SCORES_CRITICAL_HOURS:
            logger.error(
                f"Scores are critically stale ({age_hours:.1f}h old, "
                f"threshold: {STALE_SCORES_CRITICAL_HOURS}h). CronJob may have failed."
            )
        elif age_hours > STALE_SCORES_WARNING_HOURS:
            logger.warning(
                f"Scores are stale ({age_hours:.1f}h old, "
                f"threshold: {STALE_SCORES_WARNING_HOURS}h). Consider checking CronJob status."
            )

    def get_scores_age(self) -> Optional[timedelta]:
        """Return the age of the current scores."""
        if self._scores_computed_at is None:
            return None

        now = datetime.now(timezone.utc)
        computed_at = self._scores_computed_at
        if computed_at.tzinfo is None:
            computed_at = computed_at.replace(tzinfo=timezone.utc)

        return now - computed_at

    def get_data(self) -> Optional[pd.DataFrame]:
        """Return the loaded scores data."""
        return self._data
