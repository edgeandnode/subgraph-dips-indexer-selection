"""
Loads pre-computed indexer scores from a JSON file on a shared PVC.

Scores are computed daily by a CronJob (cronjobs/compute_scores/) and written
to a JSON file. IISA reads these scores on startup using DataManager.load_scores().
"""

import json
import logging
import os
from dataclasses import dataclass
from datetime import datetime, timedelta, timezone
from typing import Optional, Tuple

import pandas as pd

__all__ = ["FileScoreLoader", "DataManager", "ScoresSnapshot"]


@dataclass(frozen=True)
class ScoresSnapshot:
    """Immutable pair of (scores DataFrame, computed_at) swapped atomically.

    Why bundle: a reader that wants a consistent view of "what scores are
    loaded and when were they computed" must see both fields from the same
    push. Two separate attribute writes during a push open a window where a
    reader can land on (new computed_at, old data) or vice-versa. A single
    reference swap (`self._snapshot = ScoresSnapshot(...)`) is atomic in
    CPython, so readers always observe both fields from the same generation.
    """

    data: Optional[pd.DataFrame]
    computed_at: Optional[datetime]


# Staleness thresholds
STALE_SCORES_WARNING_HOURS = 48
STALE_SCORES_CRITICAL_HOURS = 168  # 7 days

logger = logging.getLogger(__name__)


SCORES_FILE_PATH = os.environ.get("SCORES_FILE_PATH", "/app/scores/indexer_scores.json")


class FileScoreLoader:
    """
    Reads pre-computed indexer scores from a JSON file on a shared PVC.

    The CronJob writes scores via RedpandaProvider.write_scores(); this class
    reads them back.
    """

    def __init__(self, scores_file_path: Optional[str] = None) -> None:
        # Read the module-level default at call time (not definition time) so
        # tests can monkeypatch `score_loader.SCORES_FILE_PATH` and have new
        # instances pick it up. Default capture at def-time would lock in
        # whatever value was present at module import.
        self._path = scores_file_path if scores_file_path is not None else SCORES_FILE_PATH

    @property
    def path(self) -> str:
        """The filesystem path this loader reads from and the push handler writes to."""
        return self._path

    def fetch_indexer_scores(self) -> Tuple[pd.DataFrame, Optional[datetime]]:
        """
        Read the scores JSON file and return a (DataFrame, computed_at) tuple.

        Returns (empty DataFrame, None) if the file doesn't exist or is unreadable.
        """
        logger.info(f"Reading pre-computed indexer scores from {self._path}")

        try:
            with open(self._path, "r") as f:
                data = json.load(f)
        except FileNotFoundError:
            logger.warning(f"Scores file not found: {self._path}")
            return pd.DataFrame(), None
        except (json.JSONDecodeError, OSError) as e:
            logger.error(f"Failed to read scores file {self._path}: {e}")
            return pd.DataFrame(), None

        if not data:
            logger.warning("Scores file is empty")
            return pd.DataFrame(), None

        df = pd.DataFrame(data)

        if "computed_at" not in df.columns:
            logger.warning("Scores file has no computed_at column")
            return df, None

        df["computed_at"] = pd.to_datetime(df["computed_at"], utc=True, errors="coerce")
        computed_at = df["computed_at"].iloc[0]
        if pd.isna(computed_at):
            computed_at = None

        logger.info(f"Loaded {len(df)} indexer scores from file (computed at {computed_at})")
        return df, computed_at


class DataManager:
    """
    Loads pre-computed indexer scores from the configured provider.

    Scores are computed daily by a CronJob and include latency regression
    coefficients, uptime, success rate, and economic security metrics.
    Accepts any object with a fetch_indexer_scores() method — currently
    FileScoreLoader.
    """

    def __init__(self, provider) -> None:
        self._provider = provider
        self._snapshot: ScoresSnapshot = ScoresSnapshot(data=None, computed_at=None)

    @property
    def snapshot(self) -> ScoresSnapshot:
        """Atomic snapshot of (data, computed_at) — read once, consistent pair."""
        return self._snapshot

    @property
    def scores_file_path(self) -> Optional[str]:
        """
        The on-disk path of the underlying file-backed provider, if any.

        Returns None if the provider has no .path attribute — e.g. a future
        non-FileScoreLoader provider that doesn't persist to a single file.
        Callers in the push path must guard for None before attempting to
        write to disk.
        """
        return getattr(self._provider, "path", None)

    def load_scores(self) -> bool:
        """
        Load pre-computed indexer scores from the configured provider.

        :return: True if scores were loaded successfully, False otherwise.
        """
        logger.info("Loading pre-computed indexer scores")

        scores_df, computed_at = self._provider.fetch_indexer_scores()

        if scores_df.empty:
            logger.warning("No pre-computed scores available")
            self._snapshot = ScoresSnapshot(data=None, computed_at=None)
            return False

        self._check_scores_staleness(computed_at)
        transformed = self._transform_scores_to_perf_history(scores_df)
        # Atomic swap — a reader concurrent with this line sees either
        # the previous snapshot or the new one, never a mix.
        self._snapshot = ScoresSnapshot(data=transformed, computed_at=computed_at)

        logger.info(f"Loaded scores for {len(transformed)} indexers")
        return True

    def transform_scores_df(self, scores_df: pd.DataFrame) -> pd.DataFrame:
        """
        Pure transform: run the same column mapping as load_scores() and
        return the result without touching self._snapshot.

        Used by the push path (POST /scores) to dry-run the transform
        before committing anything to disk or in-memory state. A transform
        failure raised here surfaces as an HTTP error with no side effects,
        so a malformed payload can never poison the cache.

        :raises ValueError: if the input DataFrame is empty.
        :raises Exception: if the transform fails (e.g. missing columns).
        """
        if scores_df.empty:
            raise ValueError("transform_scores_df called with empty DataFrame")
        return self._transform_scores_to_perf_history(scores_df)

    def commit_scores(
        self,
        transformed_df: pd.DataFrame,
        computed_at: Optional[datetime],
    ) -> None:
        """
        Commit an already-transformed scores DataFrame to in-memory state.

        Callers must have run transform_scores_df() on the raw input first;
        this method performs the atomic snapshot swap and logs staleness.
        Split out from load_scores_from_df so the push handler can
        validate-then-write-disk-then-commit in three distinct steps
        instead of two coupled ones.
        """
        self._check_scores_staleness(computed_at)
        # Atomic swap — see ScoresSnapshot docstring.
        self._snapshot = ScoresSnapshot(data=transformed_df, computed_at=computed_at)
        logger.info("Committed %d scores to in-memory state", len(transformed_df))

    def load_scores_from_df(
        self,
        scores_df: pd.DataFrame,
        computed_at: Optional[datetime],
    ) -> bool:
        """
        Load scores from an already-parsed DataFrame in one step.

        Facade over transform_scores_df + commit_scores. Kept for the
        non-push path and for backwards compatibility with existing
        callers that want the transform-and-commit semantics without
        the intermediate dry-run. The push handler uses the split
        methods directly.

        :return: True if scores were accepted, False if the DataFrame was empty.
        """
        if scores_df.empty:
            logger.warning("load_scores_from_df called with empty DataFrame")
            self._snapshot = ScoresSnapshot(data=None, computed_at=None)
            return False

        transformed = self.transform_scores_df(scores_df)
        self.commit_scores(transformed, computed_at)
        return True

    def _transform_scores_to_perf_history(self, scores_df: pd.DataFrame) -> pd.DataFrame:
        """
        Transform indexer_scores table format to IndexerSelector-compatible format.

        Column mapping:
        - lat_coefficient_upper_bound -> "Latency Coefficient + Error Confidence Interval"
        - uptime_score (0-1) -> "% up_x" (0-100)
        - success_rate -> "average_status"
        - dst_lat, dst_lon -> "destination_loc"
        - norm_stake_to_fees -> "norm_stake_to_fees"
        """
        df = scores_df.copy()

        # TODO: Refactor IndexerSelector to use CronJob column names directly
        df = df.rename(
            columns={
                "lat_coefficient_upper_bound": "Latency Coefficient + Error Confidence Interval",
                "success_rate": "average_status",
                "lat_normalized_score": "norm_lat_lin_reg_coefficient",
            }
        )

        if "uptime_score" in df.columns:
            df["% up_x"] = df["uptime_score"] * 100

        if "dst_lat" in df.columns and "dst_lon" in df.columns:
            df["destination_loc"] = (
                df["dst_lat"].fillna(0).astype(str) + "," + df["dst_lon"].fillna(0).astype(str)
            )

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
        snap = self._snapshot
        if snap.computed_at is None:
            return None

        now = datetime.now(timezone.utc)
        computed_at = snap.computed_at
        if computed_at.tzinfo is None:
            computed_at = computed_at.replace(tzinfo=timezone.utc)

        return now - computed_at

    def get_data(self) -> Optional[pd.DataFrame]:
        """Return the loaded scores data."""
        return self._snapshot.data
