"""
BigQuery provider for reading pre-computed indexer scores.
"""

import logging
import socket
from datetime import datetime
from typing import NewType, Optional, Tuple

import pandas as pd
from bigframes import pandas as bpd
from tenacity import (
    retry,
    retry_if_exception_type,
    stop_after_attempt,
    wait_exponential,
)

QueryStr = NewType("QueryStr", str)

logger = logging.getLogger(__name__)


class BigQueryProvider:
    """Reads pre-computed indexer scores from BigQuery."""

    def __init__(self, project: str, location: str) -> None:
        bpd.options.bigquery.project = project
        bpd.options.bigquery.location = location
        bpd.options.display.progress_bar = None

    @retry(
        retry=retry_if_exception_type((ConnectionError, socket.timeout)),
        stop=stop_after_attempt(3),
        wait=wait_exponential(multiplier=1, max=60),
        reraise=True,
    )
    def _read_gbq_dataframe(self, query: QueryStr) -> pd.DataFrame:
        return bpd.read_gbq(query).to_pandas()

    def fetch_indexer_scores(
        self, dataset: str = "iisa_data_for_dips"
    ) -> Tuple[pd.DataFrame, Optional[datetime]]:
        """
        Fetch pre-computed indexer scores from the indexer_scores table.

        Returns ~60 rows (one per indexer) computed daily by CronJob.

        :param dataset: The BigQuery dataset containing the indexer_scores table.
        :return: Tuple of (DataFrame with scores, timestamp when computed).
        """
        logger.info("Fetching pre-computed indexer scores from BigQuery")

        project = bpd.options.bigquery.project

        query = QueryStr(f"""
            SELECT *
            FROM `{project}.{dataset}.indexer_scores`
            WHERE computed_at = (
                SELECT MAX(computed_at)
                FROM `{project}.{dataset}.indexer_scores`
            )
        """)

        dataframe = self._read_gbq_dataframe(query)

        if dataframe.empty:
            logger.warning("No scores found in indexer_scores table")
            return dataframe, None

        computed_at = pd.to_datetime(dataframe["computed_at"].iloc[0])
        logger.info(f"Fetched {len(dataframe)} indexer scores (computed at {computed_at})")

        return dataframe, computed_at
