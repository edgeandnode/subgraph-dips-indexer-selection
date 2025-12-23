"""
The "Google BigQuery" provider.
"""

import logging
import socket
from datetime import date
from textwrap import dedent
from typing import NewType, cast

import pandas as pd
import pandera as pa
from bigframes import pandas as bpd
from pandera.typing import DataFrame, Index, Series
from tenacity import (
    retry,
    retry_if_exception_type,
    stop_after_attempt,
    wait_exponential,
)

from .time import DateStr, TimestampStr
from .typing import (
    ArrowDate32Field,
    DeploymentIdField,
    HttpUrlField,
    IndexerIdField,
    QueryIdField,
)

QueryStr = NewType("QueryStr", str)


# Module-level logger
logger = logging.getLogger(__name__)


class InitialQuerySchema(pa.DataFrameModel):
    """
    Schema for the initial query results data frame.

    See BigQueryProvider.fetch_initial_query_results for more information.
    """

    deployment_hash: Series[str] = DeploymentIdField()
    indexer: Series[str] = IndexerIdField()
    num_rows: Series[int] = pa.Field(ge=0)


class CombinedQuerySchema(pa.DataFrameModel):
    """
    Schema for the combined query results dataframe.

    See BigQueryProvider.fetch_combined_query_results for more information.
    """

    query_id: Series[str] = QueryIdField()
    deployment_hash: Series[str] = DeploymentIdField()
    fee: Series[float] = pa.Field()
    # timestamp: Series[pd.Timestamp] = pa.Field()
    blocks_behind: Series[int] = pa.Field(ge=0)
    response_time_ms: Series[int] = pa.Field(ge=0)
    indexer: Series[str] = IndexerIdField()
    status: Series[str] = pa.Field()
    day_partition: Series[pd.ArrowDtype] = ArrowDate32Field()
    subgraph_network: Series[str] = pa.Field(str_length={"min_value": 1})
    url: Series[str] = HttpUrlField()


class StakeToFeesSchema(pa.DataFrameModel):
    """
    Schema for the stake-to-fees ratio dataframe.
    """

    indexer: Index[str] = IndexerIdField()
    recent_slashable_stake: Series[float] = pa.Field(ge=0)
    total_query_fees_sum: Series[float] = pa.Field(ge=0)
    stake_to_fees: Series[float] = pa.Field(ge=0)


InitialQueryDataFrame = DataFrame[InitialQuerySchema]
CombinedQueryDataFrame = DataFrame[CombinedQuerySchema]
StakeToFeesDataFrame = DataFrame[StakeToFeesSchema]


class BigQueryProvider:
    """A class that provides read access to Google BigQuery"""

    def __init__(self, project: str, location: str) -> None:
        # Configure the Google BigQuery dataframes project and location
        bpd.options.bigquery.project = project
        bpd.options.bigquery.location = location
        bpd.options.display.progress_bar = None

    @retry(
        retry=retry_if_exception_type((ConnectionError, socket.timeout)),
        stop=stop_after_attempt(10),
        wait=wait_exponential(multiplier=1, max=60),
        reraise=True,  # (Default) After set number of attempts the decorator will re-raise the issue further up.
    )
    def _read_gbq_dataframe(self, query: QueryStr) -> DataFrame:
        """
        Execute a read query on Google BigQuery and return the results as a pandas DataFrame.

        If an error occurs during the query execution, the method will retry the query up to 10 times with an
        exponential backoff strategy.

        .. note::
            This method uses the bigframes.pandas.read_gbq function to execute the query. It relies on
            Application Default Credentials (ADC) for authentication, primarily using the
            GOOGLE_APPLICATION_CREDENTIALS environment variable if set. This variable should point to
            the JSON file containing the service account key.

        :param query: SQL query string to be executed on Google BigQuery.
        :return: DataFrame containing the query results.
        """
        return cast(DataFrame, bpd.read_gbq(query).to_pandas())

    @pa.check_types
    def fetch_initial_query_results(
        self, start_date: date, num_days: int
    ) -> InitialQueryDataFrame:
        """
        Fetch the initial query results.
        This query produces a table on the order of about 20,000 rows. With 3 columns.
        This query has no need for pagination.

        :param start_date: The start date for the query range.
        :param num_days: The number of days to include in the query range.
        :return: A DataFrame containing the initial query results.
        """
        # Format the start date as a %Y-%m-%d string
        start = DateStr(start_date.strftime("%Y-%m-%d"))

        logger.debug(
            "Fetching initial query results",
            extra={"start_date": start, "num_days": num_days},
        )

        query = _get_initial_query(start, num_days)
        dataframe = self._read_gbq_dataframe(query)

        if not dataframe.empty:
            dataframe.sort_values(by="num_rows", ascending=False, inplace=True)

        logger.debug(
            f"Fetched initial query results ({dataframe.shape[0]})",
            extra={"rows": dataframe.shape[0]},
        )

        return cast(InitialQueryDataFrame, dataframe)

    @pa.check_types
    def fetch_combined_query_results(
        self, start_date: date, num_days: int, rows_to_use: int
    ) -> CombinedQueryDataFrame:
        """
        Fetch the combined query results, handling large datasets by writing results to a BigQuery table.
        This query produces a table on the order of about 20,000,000 rows. With 11 columns.
        This query requires pagination.

        This function constructs a query using `_get_combined_query`, executes it, writes the
        results to a destination table, and retrieves the data into a Pandas DataFrame.

        :param start_date: The start date for the query range.
        :param num_days: The number of days to include in the query range.
        :param rows_to_use: The maximum number of rows to retrieve per deployment_hash and indexer combination.
        :return: A DataFrame containing the combined query results.
        """
        # Format the start date as a %Y-%m-%d string
        start = DateStr(start_date.strftime("%Y-%m-%d"))

        logger.debug(
            "Fetching combined query results",
            extra={
                "start_date": start,
                "num_days": num_days,
                "rows_to_use": rows_to_use,
            },
        )

        # Generate the SQL query
        query = _get_combined_query(start, num_days, rows_to_use)

        # Table to store the intermediate data
        destination_table = "graph-mainnet.iisa_data_for_dips.get_combined_query_data"

        try:
            # Run the SQL query in BigQuery, store the result in the destination table
            intermediate_dataframe = bpd.read_gbq(query)  # Keep as bigframes DataFrame
            intermediate_dataframe.to_gbq(destination_table, if_exists="replace")

            logger.debug("Data written to intermediate table, beginning table read.")

            # Batch sizes to try. This is a fallback mechanism incase a batch is unsuccessful.
            batch_sizes = [10_000_000, 5_000_000, 2_500_000, 1_000_000]
            dataframe = pd.DataFrame()
            successful = False

            # Try different batch sizes if issues arise
            for batch_size in batch_sizes:
                try:
                    logger.debug(
                        f"Attempting to read data with batch size: {batch_size}"
                    )

                    offset = 0
                    dataframe = pd.DataFrame()

                    while True:
                        # Read data in chunks using LIMIT and OFFSET
                        paginated_query = QueryStr(
                            f"""
                            SELECT *
                            FROM `{destination_table}`
                            LIMIT {batch_size}
                            OFFSET {offset}
                            """
                        )
                        batch_dataframe = self._read_gbq_dataframe(paginated_query)

                        if batch_dataframe.empty:
                            logger.debug(
                                "No more data to fetch, exiting pagination loop."
                            )
                            break

                        logger.debug(
                            f"Fetched {batch_dataframe.shape[0]} rows from offset {offset}."
                        )

                        dataframe = pd.concat(
                            [dataframe, batch_dataframe], ignore_index=True
                        )
                        offset += batch_size

                    # If we successfully reach here, break out of the batch size loop
                    successful = True
                    break

                except Exception as batch_error:
                    logger.warning(
                        f"Batch size {batch_size} failed, retrying with smaller batch size. Error: {batch_error}",
                        exc_info=True,
                    )

                    # Try the next batch size
                    continue

            # If no batch size worked, raise an error
            if not successful:
                raise RuntimeError("Failed to fetch data with any batch size.")

            # Post-processing the dataframe
            if not dataframe.empty:
                # Drop rows with missing values in the "url" column
                dataframe.dropna(subset=["url"], inplace=True)

                # Add trailing slash if not present
                dataframe["url"] = dataframe["url"].apply(
                    lambda url: url if url.endswith("/") else url + "/"
                )

            logger.debug(
                f"Fetched combined query results ({dataframe.shape[0]})",
                extra={"rows": dataframe.shape[0]},
            )

            return cast(CombinedQueryDataFrame, dataframe)

        except Exception as e:
            logger.error("Failed to fetch combined query results.", exc_info=True)
            raise e

    @pa.check_types
    def fetch_initial_stake_to_fees(
        self, start_ts: TimestampStr
    ) -> StakeToFeesDataFrame:
        """
        Get the initial stake to fees query.
        This query produces a table on the order of about 100 rows. With 4 columns.
        This query has no need for pagination.

        :param start_ts: The starting timestamp for the query.
        :return: A DataFrame containing the stake-to-fees query results.
        """
        logger.debug(
            "Fetching initial stake-to-fees query", extra={"start_ts": start_ts}
        )

        query = _get_initial_stake_to_fees_query(start_ts)
        dataframe = self._read_gbq_dataframe(query)

        # Set the indexer column as the index
        dataframe.set_index("indexer", inplace=True)

        logger.debug(
            f"Fetched initial stake-to-fees query ({dataframe.shape[0]})",
            extra={"rows": dataframe.shape[0]},
        )

        return cast(StakeToFeesDataFrame, dataframe)


def _get_combined_query(
    start_date: DateStr, num_days: int, rows_to_use: int
) -> QueryStr:
    """
    Construct a SQL query to fetch detailed data from multiple tables.

    This function generates a complex SQL query that combines data from production_metrics,
    indexer_dimensions, and metrics_indexer_attempts tables. It includes subquery logic
    to handle deployment networks, indexer networks, and data sampling.

    :param start_date: The start date for the query range.
    :param num_days: The number of days to include in the query range.
    :param rows_to_use: The maximum number of rows to retrieve per deployment_hash and indexer combination.
    :return: A SQL query string that selects and combines data from multiple tables,
             applying various filters and transformations.
    """
    return QueryStr(
        dedent(f"""\
        WITH production_metrics_gateway_subgraph_queries AS (
            WITH initial_data AS (
                SELECT
                    day_timestamp AS day_partition,
                    subgraph_deployment_ipfs_hash AS deployment_hash,
                    subgraph_chain_indexed AS subgraph_network,
                    subgraph_deployment_chain AS indexer_network
                FROM production_metrics.prod_metrics_gateway_subgraph_queries
                WHERE subgraph_deployment_ipfs_hash IS NOT NULL
                AND subgraph_chain_indexed IS NOT NULL
                AND subgraph_deployment_chain IS NOT NULL
            ),
            non_dupe_data AS (
                SELECT DISTINCT * FROM initial_data
            ),
            mode_subgraph_networks AS (
                SELECT
                    deployment_hash,
                    subgraph_network,
                    COUNT(subgraph_network) AS freq
                FROM non_dupe_data
                GROUP BY deployment_hash, subgraph_network
            ),
            aggregated_data AS (
                SELECT
                    n.deployment_hash,
                    ARRAY_AGG(n.indexer_network) AS indexer_network_list,
                    ARRAY_AGG(DISTINCT n.subgraph_network) AS subgraph_network_list,
                    COUNT(DISTINCT n.indexer_network) AS number_of_unique_indexer_networks,
                    COUNT(n.indexer_network) AS number_of_indexer_networks,
                    ARRAY_AGG(s.subgraph_network ORDER BY s.freq DESC LIMIT 1)[OFFSET(0)] AS mode_subgraph_network
                FROM non_dupe_data n
                LEFT JOIN mode_subgraph_networks s
                ON n.deployment_hash = s.deployment_hash
                GROUP BY n.deployment_hash
            )
            SELECT
                deployment_hash,
                CASE
                    WHEN ARRAY_LENGTH(indexer_network_list) = 1 THEN indexer_network_list[OFFSET(0)]
                    ELSE 'arbitrum'
                END AS indexer_network,
                CASE
                    WHEN ARRAY_LENGTH(subgraph_network_list) = 1 THEN subgraph_network_list[OFFSET(0)]
                    ELSE mode_subgraph_network
                END AS subgraph_network
            FROM aggregated_data
            WHERE deployment_hash IS NOT NULL
            AND deployment_hash <> ''
            ORDER BY number_of_unique_indexer_networks DESC
        ),

        combined_indexer_dimensions AS (
            WITH indexer_dimensions AS (
                SELECT
                    day AS day_partition,
                    indexer_wallet AS indexer,
                    indexer_url AS url,
                    'mainnet-gateway' AS indexer_network
                FROM internal_metrics.indexer_dimensions_daily
                WHERE day BETWEEN '{start_date}' AND DATE_ADD('{start_date}', INTERVAL {num_days} DAY)
            ),
            indexer_dimensions_arbitrum AS (
                SELECT
                    day AS day_partition,
                    indexer_wallet AS indexer,
                    indexer_url AS url,
                    'mainnet-thegraph-arbitrum' AS indexer_network
                FROM internal_metrics.indexer_dimensions_arbitrum_daily
                WHERE day BETWEEN '{start_date}' AND DATE_ADD('{start_date}', INTERVAL {num_days} DAY)
            ),
            combined_data AS (
                SELECT * FROM indexer_dimensions
                UNION ALL
                SELECT * FROM indexer_dimensions_arbitrum
            )
            SELECT
                day_partition,
                indexer,
                url,
                CASE
                    WHEN indexer_network = 'mainnet-thegraph-arbitrum' THEN 'arbitrum'
                    WHEN indexer_network = 'mainnet-gateway' THEN 'mainnet'
                END AS indexer_network
            FROM combined_data
            WHERE indexer IS NOT NULL AND url IS NOT NULL
            GROUP BY day_partition, indexer, url, indexer_network
            ORDER BY day_partition
        ),

        metrics_indexer_attempts AS (
            WITH BasicFilter AS (
                SELECT
                    query_id,
                    deployment AS deployment_hash,
                    query_fee AS fee,
                    query_ts AS timestamp,
                    CAST(blocks_behind AS INT64) AS blocks_behind,
                    SAFE_CAST(response_time_ms AS INT64) AS response_time_ms,
                    indexer,
                    status,
                    day_partition,
                    RAND() as rnd
                FROM internal_metrics.metrics_indexer_attempts
                WHERE day_partition BETWEEN '{start_date}' AND DATE_ADD('{start_date}', INTERVAL {num_days} DAY)
                AND deployment IN (SELECT deployment_hash FROM production_metrics_gateway_subgraph_queries)
            ),
            FilteredRows AS (
                SELECT
                    *,
                    ROW_NUMBER() OVER (PARTITION BY deployment_hash, indexer ORDER BY rnd) as row_num
                FROM BasicFilter
            )
            SELECT
                query_id,
                deployment_hash,
                fee,
                timestamp,
                blocks_behind,
                response_time_ms,
                indexer,
                status,
                day_partition
            FROM FilteredRows
            WHERE row_num <= {rows_to_use}
        )

        SELECT
            m.query_id,
            m.deployment_hash,
            m.fee,
            m.timestamp,
            m.blocks_behind,
            m.response_time_ms,
            m.indexer,
            m.status,
            m.day_partition,
            pm.subgraph_network,
            c.url
        FROM metrics_indexer_attempts m
        LEFT JOIN production_metrics_gateway_subgraph_queries pm
        ON m.deployment_hash = pm.deployment_hash
        LEFT JOIN combined_indexer_dimensions c
        ON m.indexer = c.indexer AND m.day_partition = c.day_partition AND pm.indexer_network = c.indexer_network
        WHERE pm.indexer_network = 'arbitrum'
        ORDER BY m.timestamp;
    """)
    )


def _get_initial_query(start_date: DateStr, num_days: int) -> QueryStr:
    """
    Construct an initial SQL query to fetch basic filter data from the metrics_indexer_attempts table.

    This function generates a SQL query that counts the number of rows for each combination of
    deployment hash and indexer within a specified date range.

    :param start_date: The start date for the query range.
    :param num_days: The number of days to include in the query range.
    :return: A SQL query string that selects deployment_hash, indexer, and num_rows,
             filtered by the specified date range.
    """
    return QueryStr(
        dedent(f"""\
        WITH BasicFilter AS (
            SELECT
                deployment AS deployment_hash,
                indexer,
                COUNT(*) AS num_rows
            FROM internal_metrics.metrics_indexer_attempts
            WHERE day_partition BETWEEN '{start_date}' AND DATE_ADD('{start_date}', INTERVAL {num_days} DAY)
            GROUP BY deployment_hash, indexer
        ),
        TotalQueries AS (
            SELECT
                deployment_hash,
                indexer,
                num_rows
            FROM BasicFilter
        )
        SELECT
            deployment_hash,
            indexer,
            num_rows
        FROM TotalQueries;
    """)
    )


def _get_initial_stake_to_fees_query(start_ts: TimestampStr) -> QueryStr:
    """
    A SQL query to calculate the stake-to-fees ratio for indexers.

    This function constructs a SQL query that computes the ratio of slashable stake
    to total query fees each indexer in the Arbitrum network has received, regardless
    of the collection status, starting from a specified timestamp. In this case the
    start_ts is a date time string num_days before the current day. This way any historical
    query fees earned outside the looked upon window does not affect an indexers'
    current stake-to-fees ratio.

    Note:
    - The query joins data from 'internal_metrics.indexer_dimensions_arbitrum' and
      'internal_metrics.metrics_indexer_attempts' tables.
    - The query filters data starting from the provided timestamp.

    :param start_ts: The starting timestamp for the query, formatted as a string.
    :return: A SQL query string that calculates stake-to-fees ratios.
    """
    return QueryStr(
        dedent(f"""\
        SELECT indexer,
            recent_slashable_stake,
            SUM(query_fees_sum) AS total_query_fees_sum,
            recent_slashable_stake / SUM(query_fees_sum) as stake_to_fees
        FROM (
            SELECT  id.indexer_wallet AS indexer,
                    id.staked_tokens - id.locked_tokens as recent_slashable_stake,
                    SUM(mia.query_fee) AS query_fees_sum
            FROM internal_metrics.indexer_dimensions_arbitrum id
            INNER JOIN internal_metrics.metrics_indexer_attempts mia ON id.indexer_wallet = mia.indexer
            WHERE TIMESTAMP(mia.day_partition) > '{start_ts}'
            GROUP BY id.indexer_wallet, id.staked_tokens - id.locked_tokens, mia.day_partition
        ) as aggregated_data
        GROUP BY indexer, recent_slashable_stake;
    """)
    )
