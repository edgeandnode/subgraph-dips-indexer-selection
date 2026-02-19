"""
RedpandaProvider: Redpanda-backed data source for the score computation CronJob.

Replaces BigQuery as the raw data source. A daily batch replay consumes the
gateway_queries topic for the 28-day regression window, applies stratified
reservoir sampling, and produces DataFrames with the same schema that
processing.py expects.

Stake data is fetched via GraphQL from the Graph Network subgraph.
Computed scores are written to a JSON file on a shared PVC instead of BigQuery.
"""

import json
import logging
import os
import random
from collections import defaultdict
from datetime import date, datetime, timedelta, timezone
from pathlib import Path
from typing import Dict, List, Optional, Tuple

import base58
import pandas as pd
import requests

from proto.gateway_queries_pb2 import ClientQueryProtobuf

logger = logging.getLogger(__name__)

# Maximum rows buffered per (deployment_hash, indexer) pair.
# Algorithm R reservoir — excess messages replace random earlier entries.
MAX_RESERVOIR_PER_PAIR = 50_000

SCORES_FILE_PATH = os.environ.get("SCORES_FILE_PATH", "/app/scores/indexer_scores.json")


# ---------------------------------------------------------------------------
# Encoding helpers
# ---------------------------------------------------------------------------


def _bytes_to_cid(b: bytes) -> str:
    """Encode a 32-byte deployment hash as a CIDv0 base58 string (Qm...)."""
    if len(b) != 32:
        return ""
    multihash = b"\x12\x20" + b  # sha2-256 function code + 32-byte length prefix
    return base58.b58encode(multihash).decode("ascii")


def _bytes_to_hex(b: bytes) -> str:
    """Encode a 20-byte Ethereum address as a lowercase 0x-prefixed hex string."""
    if len(b) != 20:
        return ""
    return "0x" + b.hex().lower()


def _map_result_to_status(result: str) -> str:
    """Map a protobuf result value to the status string processing.py expects."""
    return "200 OK" if result == "success" else result


# ---------------------------------------------------------------------------
# RedpandaProvider
# ---------------------------------------------------------------------------


class RedpandaProvider:
    """
    Duck-typed replacement for BigQueryClient.

    Implements the same interface used by compute_all_scores:
      fetch_initial_query_results, fetch_combined_query_results,
      fetch_stake_to_fees, write_scores, scores_exist_for_today.

    The Kafka replay is performed once per day; both fetch methods share a
    single in-memory cache built during _stream_and_cache.
    """

    def __init__(self) -> None:
        self._bootstrap_servers = os.environ.get("REDPANDA_BOOTSTRAP_SERVERS", "")
        self._topic = os.environ.get("REDPANDA_TOPIC", "gateway_queries")
        self._graph_network_url = os.environ.get("GRAPH_NETWORK_SUBGRAPH_URL", "")
        self._scores_path = SCORES_FILE_PATH

        # Reservoir cache — populated by _stream_and_cache
        self._count_cache: Dict[Tuple[str, str], int] = {}
        self._row_cache_df: Optional[pd.DataFrame] = None
        self._cache_start_date: Optional[date] = None
        self._cache_num_days: Optional[int] = None

    # -----------------------------------------------------------------------
    # Public interface
    # -----------------------------------------------------------------------

    def fetch_initial_query_results(self, start_date: date, num_days: int) -> pd.DataFrame:
        """
        Return row counts per (deployment_hash, indexer) pair.

        Matches BigQueryClient.fetch_initial_query_results output schema:
        columns [deployment_hash, indexer, num_rows].
        """
        self._ensure_cache(start_date, num_days)

        rows = [
            {"deployment_hash": dep, "indexer": idx, "num_rows": count}
            for (dep, idx), count in self._count_cache.items()
        ]
        df = pd.DataFrame(rows, columns=["deployment_hash", "indexer", "num_rows"])
        if not df.empty:
            df.sort_values(by="num_rows", ascending=False, inplace=True, ignore_index=True)

        memory_mb = df.memory_usage(deep=True).sum() / (1024 * 1024)
        logger.info(f"Initial query results from Redpanda: {len(df)} pairs ({memory_mb:.1f} MB)")
        return df

    def fetch_combined_query_results(
        self, start_date: date, num_days: int, rows_to_use: int
    ) -> pd.DataFrame:
        """
        Return sampled query rows for regression analysis.

        Caps each (deployment_hash, indexer) pair to rows_to_use rows.
        Because the reservoir rows are already drawn uniformly at random
        (Algorithm R), truncation is equivalent to further sampling.

        Matches BigQueryClient.fetch_combined_query_results output schema:
        columns [query_id, deployment_hash, fee, timestamp, blocks_behind,
                 response_time_ms, indexer, status, day_partition,
                 subgraph_network, url].
        """
        self._ensure_cache(start_date, num_days)

        if self._row_cache_df is None or self._row_cache_df.empty:
            logger.warning("Row cache is empty — no data in the specified Redpanda window")
            return pd.DataFrame(
                columns=[
                    "query_id", "deployment_hash", "fee", "timestamp",
                    "blocks_behind", "response_time_ms", "indexer", "status",
                    "day_partition", "subgraph_network", "url",
                ]
            )

        df = (
            self._row_cache_df
            .groupby(["deployment_hash", "indexer"], group_keys=False)
            .apply(lambda x: x.iloc[:rows_to_use])
            .reset_index(drop=True)
        )

        memory_mb = df.memory_usage(deep=True).sum() / (1024 * 1024)
        logger.info(f"Combined query results from Redpanda: {len(df)} rows ({memory_mb:.1f} MB)")
        return df

    def fetch_stake_to_fees(self, start_ts: str) -> pd.DataFrame:
        """
        Fetch stake data from the Graph Network subgraph via GraphQL.

        Returns a DataFrame indexed by 'indexer' with a 'stake_to_fees' column.
        stake_to_fees is NaN because query fee data is not available from the
        subgraph; processing.py handles missing stake data gracefully with NaN.

        If GRAPH_NETWORK_SUBGRAPH_URL is not set, returns an empty DataFrame.
        """
        if not self._graph_network_url:
            logger.warning(
                "GRAPH_NETWORK_SUBGRAPH_URL not set — returning empty stake data. "
                "stake_to_fees will be NaN for all indexers."
            )
            return pd.DataFrame(columns=["stake_to_fees"])

        logger.info(f"Fetching stake data from {self._graph_network_url}")
        try:
            indexers = self._paginate_graphql_indexers()
        except Exception:
            logger.exception("Failed to fetch stake data from subgraph")
            return pd.DataFrame(columns=["stake_to_fees"])

        if not indexers:
            logger.warning("Subgraph returned no indexer records")
            return pd.DataFrame(columns=["stake_to_fees"])

        df = pd.DataFrame(indexers)
        df["recent_slashable_stake"] = (
            df["stakedTokens"].astype(float) - df["lockedTokens"].astype(float)
        )
        # Query fees are not available from the subgraph, so stake_to_fees is NaN.
        df["stake_to_fees"] = float("nan")
        df = df.rename(columns={"id": "indexer"})[["indexer", "stake_to_fees"]]
        df.set_index("indexer", inplace=True)

        logger.info(f"Fetched stake data for {len(df)} indexers")
        return df

    def write_scores(self, scores_df: pd.DataFrame) -> None:
        """
        Serialise the scores DataFrame to a JSON file atomically.

        Uses a write-to-tmp-then-rename pattern so the HTTP service never
        reads a partially-written file.
        """
        scores_path = Path(self._scores_path)
        scores_path.parent.mkdir(parents=True, exist_ok=True)
        tmp_path = str(scores_path) + ".tmp"

        # Serialise datetimes to ISO strings for round-trip fidelity.
        data = scores_df.copy()
        for col in data.select_dtypes(include=["datetime64[ns, UTC]", "datetime64[ns]"]).columns:
            data[col] = data[col].dt.strftime("%Y-%m-%dT%H:%M:%S%z")

        data.to_json(tmp_path, orient="records", date_format="iso", date_unit="s")
        os.replace(tmp_path, str(scores_path))

        memory_mb = scores_df.memory_usage(deep=True).sum() / (1024 * 1024)
        logger.info(
            f"Wrote {len(scores_df)} scores ({memory_mb:.2f} MB) to {self._scores_path}"
        )

    def scores_exist_for_today(self) -> bool:
        """
        Return True if the scores file already contains today's computation.

        Loads the full JSON file and checks the first record's computed_at field.
        """
        try:
            with open(self._scores_path, "r") as f:
                data = json.load(f)
            if not data:
                return False
            computed_at_str = data[0].get("computed_at", "")
            if not computed_at_str:
                return False
            computed_at = pd.to_datetime(computed_at_str, utc=True)
            return computed_at.date() == datetime.now(timezone.utc).date()
        except FileNotFoundError:
            return False
        except Exception as e:
            logger.warning(f"Could not check scores file: {e}")
            return False

    # -----------------------------------------------------------------------
    # Internal: cache management
    # -----------------------------------------------------------------------

    def _ensure_cache(self, start_date: date, num_days: int) -> None:
        """Trigger a Kafka replay only if the cache doesn't cover this window."""
        if (
            self._row_cache_df is not None
            and self._cache_start_date == start_date
            and self._cache_num_days == num_days
        ):
            return
        self._stream_and_cache(start_date, num_days)

    def _stream_and_cache(self, start_date: date, num_days: int) -> None:
        """
        Replay the gateway_queries topic for the given window.

        Seeks to start_date using offset-for-timestamp, polls until
        start_date + num_days, and applies Algorithm R reservoir sampling
        per (deployment_hash, indexer) pair.

        After completion, populates:
          self._count_cache  — true per-pair message counts (for adjust_rows)
          self._row_cache_df — reservoir DataFrame (for fetch_combined_query_results)
        """
        from confluent_kafka import Consumer, TopicPartition, KafkaException

        start_dt = datetime.combine(start_date, datetime.min.time()).replace(tzinfo=timezone.utc)
        end_dt = start_dt + timedelta(days=num_days)
        start_ts_ms = int(start_dt.timestamp() * 1000)
        end_ts_ms = int(end_dt.timestamp() * 1000)

        logger.info(
            f"Starting Redpanda replay: topic={self._topic}, "
            f"window={start_date} to {end_dt.date()}"
        )

        consumer = Consumer(
            {
                "bootstrap.servers": self._bootstrap_servers,
                # group.id is required by confluent-kafka but unused — we call
                # assign() for manual partition assignment, bypassing group
                # coordination entirely.
                "group.id": "iisa-score-computation-replay",
                "enable.auto.commit": False,
            }
        )

        try:
            self._consume(consumer, start_ts_ms, end_ts_ms)
        finally:
            consumer.close()

        logger.info(
            f"Replay complete: {sum(self._count_cache.values())} total attempts "
            f"across {len(self._count_cache)} (deployment, indexer) pairs"
        )
        self._cache_start_date = start_date
        self._cache_num_days = num_days

    def _consume(self, consumer, start_ts_ms: int, end_ts_ms: int) -> None:
        """Drive the consumer loop and populate internal caches."""
        from confluent_kafka import TopicPartition, KafkaException, OFFSET_INVALID

        # Discover partitions.
        meta = consumer.list_topics(self._topic, timeout=30)
        if self._topic not in meta.topics:
            raise RuntimeError(f"Topic '{self._topic}' not found in Redpanda")

        partition_ids = list(meta.topics[self._topic].partitions.keys())
        seek_tps = [TopicPartition(self._topic, pid, start_ts_ms) for pid in partition_ids]

        # Resolve timestamp → offset for each partition.
        resolved = consumer.offsets_for_times(seek_tps, timeout=30)
        valid = [
            TopicPartition(tp.topic, tp.partition, tp.offset)
            for tp in resolved
            if tp.offset >= 0  # -1001 (OFFSET_INVALID) means no message at that time
        ]

        if not valid:
            logger.warning(
                "No Redpanda messages found for the requested time window — "
                f"start_ts_ms={start_ts_ms}"
            )
            self._count_cache = {}
            self._row_cache_df = _empty_combined_df()
            return

        consumer.assign(valid)

        # Reservoir state: {(deployment_hash, indexer): list of row dicts}
        reservoirs: Dict[Tuple[str, str], List[dict]] = defaultdict(list)
        counts: Dict[Tuple[str, str], int] = defaultdict(int)

        total_messages = 0
        total_attempts = 0
        skipped_messages = 0
        consecutive_timeouts = 0
        max_consecutive_timeouts = 3  # 3 × 30 s = 90 s idle before stopping

        while True:
            msg = consumer.poll(timeout=30.0)

            if msg is None:
                consecutive_timeouts += 1
                logger.debug(
                    f"Poll timeout ({consecutive_timeouts}/{max_consecutive_timeouts})"
                )
                if consecutive_timeouts >= max_consecutive_timeouts:
                    logger.info("Reached end of available data (consecutive poll timeouts)")
                    break
                continue

            consecutive_timeouts = 0

            if msg.error():
                logger.error(f"Kafka error: {msg.error()}")
                continue

            ts_type, ts_ms = msg.timestamp()
            if ts_ms < 0:
                # No timestamp — use current time as approximation and continue.
                ts_ms = int(datetime.now(timezone.utc).timestamp() * 1000)

            if ts_ms > end_ts_ms:
                logger.info("Reached end of replay window (message timestamp past end_ts)")
                break

            total_messages += 1

            try:
                client_query = ClientQueryProtobuf.FromString(msg.value())
            except Exception as e:
                skipped_messages += 1
                logger.debug(f"Failed to parse message: {e}")
                continue

            timestamp = pd.Timestamp(ts_ms, unit="ms", tz="UTC")
            day_partition = timestamp.date()

            for attempt in client_query.indexer_queries:
                deployment_hash = _bytes_to_cid(attempt.deployment)
                indexer = _bytes_to_hex(attempt.indexer)

                if not deployment_hash or not indexer or not attempt.url:
                    continue  # Skip malformed attempts

                total_attempts += 1

                url = attempt.url if attempt.url.endswith("/") else attempt.url + "/"
                row = {
                    "query_id": client_query.query_id,
                    "deployment_hash": deployment_hash,
                    "fee": attempt.fee_grt,
                    "timestamp": timestamp,
                    "blocks_behind": attempt.blocks_behind,
                    "response_time_ms": attempt.response_time_ms,
                    "indexer": indexer,
                    "status": _map_result_to_status(attempt.result),
                    "day_partition": day_partition,
                    "subgraph_network": attempt.indexed_chain,
                    "url": url,
                }

                key = (deployment_hash, indexer)
                n = counts[key]

                if n < MAX_RESERVOIR_PER_PAIR:
                    reservoirs[key].append(row)
                else:
                    j = random.randint(0, n - 1)
                    if j < MAX_RESERVOIR_PER_PAIR:
                        reservoirs[key][j] = row

                counts[key] += 1

            if total_messages % 100_000 == 0:
                logger.info(
                    f"  {total_messages:,} messages consumed, "
                    f"{total_attempts:,} attempts buffered"
                )

        logger.info(
            f"Consume loop done: {total_messages:,} messages, "
            f"{total_attempts:,} attempts, {skipped_messages} parse errors"
        )

        self._count_cache = dict(counts)

        all_rows: List[dict] = []
        for rows in reservoirs.values():
            all_rows.extend(rows)

        if all_rows:
            self._row_cache_df = pd.DataFrame(all_rows)
        else:
            self._row_cache_df = _empty_combined_df()

    # -----------------------------------------------------------------------
    # Internal: GraphQL pagination
    # -----------------------------------------------------------------------

    def _paginate_graphql_indexers(self) -> List[dict]:
        """Paginate through all indexers from the Graph Network subgraph."""
        query = """
        query($first: Int!, $skip: Int!) {
          indexers(first: $first, skip: $skip) {
            id
            stakedTokens
            lockedTokens
          }
        }
        """
        page_size = 1000
        skip = 0
        all_indexers: List[dict] = []

        while True:
            response = requests.post(
                self._graph_network_url,
                json={"query": query, "variables": {"first": page_size, "skip": skip}},
                timeout=30,
            )
            response.raise_for_status()
            data = response.json()

            if "errors" in data:
                raise RuntimeError(f"GraphQL errors: {data['errors']}")

            page = data.get("data", {}).get("indexers", [])
            if not page:
                break

            all_indexers.extend(page)
            if len(page) < page_size:
                break
            skip += page_size

        return all_indexers


# ---------------------------------------------------------------------------
# Helpers
# ---------------------------------------------------------------------------


def _empty_combined_df() -> pd.DataFrame:
    return pd.DataFrame(
        columns=[
            "query_id", "deployment_hash", "fee", "timestamp",
            "blocks_behind", "response_time_ms", "indexer", "status",
            "day_partition", "subgraph_network", "url",
        ]
    )
