"""
BigQuery client for the score computation job.

Handles reading raw data and writing computed scores.
"""

import logging
import os
import socket
import time
from concurrent.futures import ThreadPoolExecutor, TimeoutError as FuturesTimeoutError
from datetime import date, datetime, timezone
from textwrap import dedent

import pandas as pd
from google.oauth2 import service_account
from tenacity import (
    retry,
    retry_if_exception_type,
    stop_after_attempt,
    wait_exponential,
)

logger = logging.getLogger(__name__)


class PermissionError(Exception):
    """Raised when the service account lacks required BigQuery permissions."""

    pass


class BigQueryClient:
    """BigQuery client for reading raw data and writing computed scores."""

    def __init__(self, project: str, dataset: str, location: str) -> None:
        self.project = project
        self.dataset = dataset
        self.location = location
        self.scores_table = f"{project}.{dataset}.indexer_scores"
        self.url_cache_table = f"{project}.{dataset}.indexer_url_cache"

        # Load service account BEFORE importing bigframes to avoid interactive auth
        creds_file = os.environ.get("GOOGLE_APPLICATION_CREDENTIALS")
        if creds_file and os.path.exists(creds_file):
            credentials = service_account.Credentials.from_service_account_file(
                creds_file,
                scopes=["https://www.googleapis.com/auth/cloud-platform"],
            )
            logger.info(f"Loaded service account from {creds_file}")
        else:
            from google.auth import default as google_auth_default
            credentials, _ = google_auth_default(
                scopes=["https://www.googleapis.com/auth/cloud-platform"]
            )
            logger.info("Using application default auth")

        # Import bigframes AFTER setting up credentials
        import bigframes
        from bigframes import pandas as bpd

        # Initialize bigframes with explicit context to avoid interactive auth
        context = bigframes.BigQueryOptions(
            credentials=credentials,
            project=project,
            location=location,
        )
        bigframes.connect(context)
        logger.info("Connected to BigQuery via bigframes")

        # Store bpd reference for use in methods
        self._bpd = bpd
        bpd.options.display.progress_bar = None

    def validate_permissions(self) -> None:
        """Validate that the service account has all required BigQuery permissions.

        Tests read access to source datasets, write access to target dataset,
        and BigQuery Storage API access. Fails fast with clear error messages.

        Raises:
            PermissionError: If any required permission is missing.
        """
        errors = []

        # Test 1: Read from internal_metrics (source data)
        logger.info("Validating permissions: testing read access to internal_metrics...")
        try:
            test_query = """
            SELECT 1 FROM `internal_metrics.metrics_indexer_attempts` LIMIT 1
            """
            self._bpd.read_gbq(test_query).to_pandas()
            logger.info("  [OK] Read access to internal_metrics.metrics_indexer_attempts")
        except Exception as e:
            error_msg = str(e)
            if "permission" in error_msg.lower() or "403" in error_msg:
                errors.append(f"Cannot read internal_metrics.metrics_indexer_attempts: {error_msg}")
                logger.error(f"  [FAIL] {errors[-1]}")
            else:
                # Non-permission error (e.g., table doesn't exist) - log but don't fail permission check
                logger.warning(f"  [WARN] Unexpected error reading internal_metrics: {error_msg}")

        # Test 2: Read from production_metrics (source data)
        logger.info("Validating permissions: testing read access to production_metrics...")
        try:
            test_query = """
            SELECT 1 FROM `production_metrics.prod_metrics_gateway_subgraph_queries` LIMIT 1
            """
            self._bpd.read_gbq(test_query).to_pandas()
            logger.info("  [OK] Read access to production_metrics.prod_metrics_gateway_subgraph_queries")
        except Exception as e:
            error_msg = str(e)
            if "permission" in error_msg.lower() or "403" in error_msg:
                errors.append(f"Cannot read production_metrics.prod_metrics_gateway_subgraph_queries: {error_msg}")
                logger.error(f"  [FAIL] {errors[-1]}")
            else:
                logger.warning(f"  [WARN] Unexpected error reading production_metrics: {error_msg}")

        # Test 3: Write to target dataset (create temp table and delete)
        logger.info("Validating permissions: testing write access to target dataset...")
        test_table = f"{self.project}.{self.dataset}._permission_test"
        try:
            create_query = f"""
            CREATE OR REPLACE TABLE `{test_table}` AS SELECT 1 as test_col
            """
            self._bpd.read_gbq(create_query)
            # Clean up
            drop_query = f"DROP TABLE IF EXISTS `{test_table}`"
            self._bpd.read_gbq(drop_query)
            logger.info(f"  [OK] Write access to {self.project}.{self.dataset}")
        except Exception as e:
            error_msg = str(e)
            if "permission" in error_msg.lower() or "403" in error_msg:
                errors.append(f"Cannot write to {self.project}.{self.dataset}: {error_msg}")
                logger.error(f"  [FAIL] {errors[-1]}")
            else:
                logger.warning(f"  [WARN] Unexpected error writing to dataset: {error_msg}")

        # Test 4: BigQuery Storage Read API (readSessionUser permission)
        # Use a larger result set to ensure Storage API is triggered (not REST fallback)
        logger.info("Validating permissions: testing BigQuery Storage Read API access...")
        try:
            storage_test_query = """
            SELECT num FROM UNNEST(GENERATE_ARRAY(1, 1000)) AS num
            """
            result = self._bpd.read_gbq(storage_test_query).to_pandas()
            if len(result) == 1000:
                logger.info("  [OK] BigQuery Storage Read API access")
            else:
                logger.warning(f"  [WARN] Storage API test returned unexpected rows: {len(result)}")
        except Exception as e:
            error_msg = str(e)
            if "permission" in error_msg.lower() or "403" in error_msg or "storage" in error_msg.lower():
                errors.append(f"Cannot use BigQuery Storage Read API: {error_msg}")
                logger.error(f"  [FAIL] {errors[-1]}")
            else:
                logger.warning(f"  [WARN] Unexpected error testing Storage API: {error_msg}")

        if errors:
            error_summary = "\n".join(f"  - {e}" for e in errors)
            raise PermissionError(
                f"Service account lacks {len(errors)} required permission(s):\n{error_summary}\n\n"
                "Required roles:\n"
                "  - roles/bigquery.dataViewer on graph-mainnet (read source data)\n"
                "  - roles/bigquery.dataEditor on iisa_data_for_dips dataset (write scores)\n"
                "  - roles/bigquery.readSessionUser on graph-mainnet (Storage API)"
            )

        logger.info("Permission validation passed")

    @retry(
        retry=retry_if_exception_type((ConnectionError, socket.timeout, TimeoutError)),
        stop=stop_after_attempt(10),
        wait=wait_exponential(multiplier=1, max=60),
        reraise=True,
    )
    def _read_gbq(self, query: str, timeout_seconds: int = 600) -> pd.DataFrame:
        """Execute a query and return results as pandas DataFrame with timeout protection."""
        with ThreadPoolExecutor(max_workers=1) as executor:
            future = executor.submit(lambda: self._bpd.read_gbq(query).to_pandas())
            try:
                return future.result(timeout=timeout_seconds)
            except FuturesTimeoutError:
                raise TimeoutError(f"BigQuery read timed out after {timeout_seconds}s")

    def scores_exist_for_today(self) -> bool:
        """Check if scores have already been computed today."""
        query = f"""
        SELECT COUNT(*) as cnt
        FROM `{self.scores_table}`
        WHERE DATE(computed_at) = CURRENT_DATE()
        """
        try:
            result = self._read_gbq(query)
            return result["cnt"].iloc[0] > 0
        except Exception as e:
            # Table might not exist yet
            logger.warning(f"Could not check for existing scores: {e}")
            return False

    def ensure_url_cache_exists(self) -> None:
        """Create the URL cache table if it doesn't exist."""
        query = f"""
        CREATE TABLE IF NOT EXISTS `{self.url_cache_table}` (
            indexer STRING NOT NULL,
            url STRING NOT NULL,
            last_seen TIMESTAMP NOT NULL
        )
        """
        try:
            self._bpd.read_gbq(query)
            logger.info(f"Ensured URL cache table exists: {self.url_cache_table}")
        except Exception:
            logger.exception(f"Failed to create URL cache table: {self.url_cache_table}")
            raise

    def update_url_cache(self, max_age_days: int = 7) -> None:
        """Incrementally update the URL cache with new data from gateway logs.

        Only updates if the cache is older than max_age_days to avoid unnecessary
        full table scans (source table is not partitioned).
        """
        # Get the last update timestamp from cache
        last_update_query = f"""
        SELECT
            COALESCE(MAX(last_seen), TIMESTAMP('1970-01-01')) as last_update,
            COUNT(*) as cache_size
        FROM `{self.url_cache_table}`
        """
        result = self._read_gbq(last_update_query)
        last_update = result["last_update"].iloc[0]
        cache_size = result["cache_size"].iloc[0]

        # Skip update if cache is fresh enough (unless empty)
        cache_age_days = (datetime.now(timezone.utc) - last_update.to_pydatetime()).days
        if cache_size > 0 and cache_age_days < max_age_days:
            logger.info(f"URL cache is fresh ({cache_age_days} days old, {cache_size} indexers), skipping update")
            return

        logger.info(f"URL cache needs update (age={cache_age_days} days, size={cache_size})")

        # Merge new data into cache
        merge_query = f"""
        MERGE `{self.url_cache_table}` AS cache
        USING (
            SELECT
                gateway_indexer_eth_address as indexer,
                ARRAY_AGG(gateway_indexer_url ORDER BY hour_timestamp DESC LIMIT 1)[OFFSET(0)] as url,
                MAX(hour_timestamp) as last_seen
            FROM `{self.project}.internal_metrics.metrics_subgraph_gateway_logs`
            WHERE hour_timestamp > TIMESTAMP('{last_update}')
              AND gateway_indexer_url IS NOT NULL AND gateway_indexer_url != ''
              AND gateway_indexer_eth_address IS NOT NULL
            GROUP BY gateway_indexer_eth_address
        ) AS new_data
        ON cache.indexer = new_data.indexer
        WHEN MATCHED AND new_data.last_seen > cache.last_seen THEN
            UPDATE SET url = new_data.url, last_seen = new_data.last_seen
        WHEN NOT MATCHED THEN
            INSERT (indexer, url, last_seen) VALUES (new_data.indexer, new_data.url, new_data.last_seen)
        """
        try:
            self._bpd.read_gbq(merge_query)
        except Exception:
            logger.exception("Failed to merge new data into URL cache")
            raise

        # Log cache stats
        stats_query = f"SELECT COUNT(*) as cnt FROM `{self.url_cache_table}`"
        stats = self._read_gbq(stats_query)
        logger.info(f"URL cache updated, total indexers: {stats['cnt'].iloc[0]}")

    def fetch_initial_query_results(self, start_date: date, num_days: int) -> pd.DataFrame:
        """
        Fetch initial query results showing row counts per deployment/indexer.

        Returns DataFrame with columns: deployment_hash, indexer, num_rows
        """
        start = start_date.strftime("%Y-%m-%d")

        query = dedent(f"""\
        WITH BasicFilter AS (
            SELECT
                deployment AS deployment_hash,
                indexer,
                COUNT(*) AS num_rows
            FROM internal_metrics.metrics_indexer_attempts
            WHERE day_partition BETWEEN '{start}' AND DATE_ADD('{start}', INTERVAL {num_days} DAY)
            GROUP BY deployment_hash, indexer
        )
        SELECT deployment_hash, indexer, num_rows
        FROM BasicFilter;
        """)

        logger.info("Fetching initial query results")
        df = self._read_gbq(query)

        if not df.empty:
            df.sort_values(by="num_rows", ascending=False, inplace=True)

        memory_mb = df.memory_usage(deep=True).sum() / (1024 * 1024)
        logger.info(f"Fetched initial query results: {len(df)} rows ({memory_mb:.1f} MB)")
        return df

    def fetch_combined_query_results(
        self, start_date: date, num_days: int, rows_to_use: int
    ) -> pd.DataFrame:
        """
        Fetch combined query results with detailed query data.

        This fetches ~20M rows with query details including response times,
        status, fees, etc.
        """
        start = start_date.strftime("%Y-%m-%d")
        destination_table = f"{self.project}.{self.dataset}.get_combined_query_data"

        query = self._get_combined_query(start, num_days, rows_to_use)

        logger.info("Running combined query and writing to intermediate table")

        # Run query and write to intermediate table
        intermediate_df = self._bpd.read_gbq(query)
        intermediate_df.to_gbq(destination_table, if_exists="replace")

        logger.info("Reading from intermediate table in batches")

        # Read back in batches
        batch_sizes = [10_000_000, 5_000_000, 2_500_000, 1_000_000]
        df = pd.DataFrame()

        for batch_size in batch_sizes:
            try:
                logger.info(f"Starting batch read with size {batch_size:,}")
                offset = 0
                df = pd.DataFrame()
                batch_num = 0
                batch_start_time = time.time()

                while True:
                    batch_num += 1
                    read_start = time.time()
                    paginated_query = f"""
                    SELECT * FROM `{destination_table}`
                    LIMIT {batch_size} OFFSET {offset}
                    """
                    batch_df = self._read_gbq(paginated_query)

                    if batch_df.empty:
                        break

                    read_elapsed = time.time() - read_start
                    total_so_far = len(df) + len(batch_df)
                    logger.info(
                        f"Batch {batch_num}: fetched {len(batch_df):,} rows in {read_elapsed:.1f}s "
                        f"(total: {total_so_far:,} rows)"
                    )
                    df = pd.concat([df, batch_df], ignore_index=True)
                    offset += batch_size

                total_elapsed = time.time() - batch_start_time
                logger.info(f"Batch read complete: {len(df):,} rows in {total_elapsed:.1f}s")
                # Success
                break

            except Exception as e:
                logger.warning(f"Batch size {batch_size:,} failed: {e}")
                continue

        if df.empty:
            raise RuntimeError("Failed to fetch combined query data")

        # Post-processing
        if not df.empty:
            rows_before = len(df)
            df.dropna(subset=["url"], inplace=True)
            rows_dropped = rows_before - len(df)
            if rows_dropped > 0:
                logger.info(f"Dropped {rows_dropped} rows with null URLs")
            df["url"] = df["url"].apply(lambda url: url if url.endswith("/") else url + "/")

        memory_mb = df.memory_usage(deep=True).sum() / (1024 * 1024)
        logger.info(f"Fetched combined query results: {len(df)} rows ({memory_mb:.1f} MB)")
        return df

    def fetch_stake_to_fees(self, start_ts: str) -> pd.DataFrame:
        """Fetch stake-to-fees data for indexers."""
        query = dedent(f"""\
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

        logger.info("Fetching stake-to-fees data")
        df = self._read_gbq(query)
        df.set_index("indexer", inplace=True)

        logger.info(f"Fetched stake-to-fees: {len(df)} indexers")
        return df

    def write_scores(self, scores_df: pd.DataFrame) -> None:
        """Write computed scores to the indexer_scores table."""
        memory_mb = scores_df.memory_usage(deep=True).sum() / (1024 * 1024)
        logger.info(f"Writing {len(scores_df)} scores ({memory_mb:.2f} MB) to {self.scores_table}")

        # Convert to bigframes and write
        bf_df = self._bpd.DataFrame(scores_df)
        bf_df.to_gbq(self.scores_table, if_exists="append")

        logger.info("Successfully wrote scores to BigQuery")

    def _get_combined_query(self, start_date: str, num_days: int, rows_to_use: int) -> str:
        """Generate the combined query SQL."""
        return dedent(f"""\
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

        indexer_url_lookup AS (
            SELECT indexer, url
            FROM `{self.url_cache_table}`
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
        LEFT JOIN indexer_url_lookup c
        ON m.indexer = c.indexer
        WHERE pm.indexer_network = 'arbitrum'
        ORDER BY m.timestamp;
        """)
