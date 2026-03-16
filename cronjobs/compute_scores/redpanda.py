"""
RedpandaProvider: Redpanda-backed data source for the score computation CronJob.

A daily batch replay consumes the gateway_queries topic for the 28-day
regression window using a two-pass architecture:

  Pass 1 (count): lightweight scan counting (deployment, indexer) pairs and
  accumulating fees.

  Pass 2 (sample): reservoir sampling with cap = rows_to_use (from adjust_rows).
  Uses raw byte keys and deferred string conversion.

Both passes consume partitions in parallel via ThreadPoolExecutor with batch
polling (consumer.consume). Partition offsets resolved in pass 1 are cached
for reuse in pass 2.

Stake data is fetched via GraphQL from the Graph Network subgraph.
Computed scores are written to a JSON file on a shared PVC.
"""

import json
import logging
import os
import random
from collections import defaultdict
from concurrent.futures import ThreadPoolExecutor
from datetime import date, datetime, timedelta, timezone
from pathlib import Path
from typing import Dict, List, Optional, Tuple

from tenacity import (
    retry,
    retry_if_exception_type,
    stop_after_attempt,
    wait_exponential,
)

# Cap parallel partition consumers to avoid unbounded thread/connection creation.
# The CronJob runs with 8 CPUs; one consumer per core saturates throughput
# without excessive Kafka connections on topics with many partitions.
MAX_PARTITION_WORKERS = int(os.environ.get("REDPANDA_MAX_WORKERS", "8"))

import base58
import pandas as pd
import requests

from gateway_queries_pb2 import ClientQueryProtobuf
from subgraph import paginate_subgraph_query

logger = logging.getLogger(__name__)

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
    Data provider for compute_all_scores.

    Implements the interface expected by compute_all_scores:
      fetch_initial_query_results, fetch_combined_query_results,
      fetch_stake_to_fees, write_scores, scores_exist_for_today.

    Uses a two-pass replay: pass 1 counts pairs and fees (cheap), pass 2
    reservoir-samples rows capped at rows_to_use (from adjust_rows).
    """

    def __init__(self) -> None:
        self._bootstrap_servers = os.environ.get("REDPANDA_BOOTSTRAP_SERVERS", "")
        self._topic = os.environ.get("REDPANDA_TOPIC", "gateway_queries")
        self.graph_network_url = os.environ.get("GRAPH_NETWORK_SUBGRAPH_URL", "")
        self._scores_path = SCORES_FILE_PATH

        # SASL authentication (optional — omit for plaintext local-network)
        self._sasl_username = os.environ.get("REDPANDA_SASL_USERNAME", "")
        self._sasl_password = os.environ.get("REDPANDA_SASL_PASSWORD", "")

        # Count cache — populated by _count_pass
        self._count_cache: Dict[Tuple[str, str], int] = {}
        self._fees_per_indexer: Dict[str, float] = {}
        self._count_cache_start_date: Optional[date] = None
        self._count_cache_num_days: Optional[int] = None

        # Row cache — populated by _sample_pass
        self._row_cache_df: Optional[pd.DataFrame] = None
        self._row_cache_start_date: Optional[date] = None
        self._row_cache_num_days: Optional[int] = None
        self._row_cache_rows_to_use: Optional[int] = None

        # Partition offsets cached between passes
        self._cached_partitions: Optional[list] = None

    def _consumer_config(self) -> dict:
        """Base librdkafka config shared by all consumers."""
        config = {
            "bootstrap.servers": self._bootstrap_servers,
            "group.id": "iisa-score-computation-replay",
            "enable.auto.commit": False,
        }
        if self._sasl_username and self._sasl_password:
            config.update({
                "security.protocol": "SASL_SSL",
                "sasl.mechanism": "SCRAM-SHA-256",
                "sasl.username": self._sasl_username,
                "sasl.password": self._sasl_password,
            })
        return config

    # -----------------------------------------------------------------------
    # Public interface
    # -----------------------------------------------------------------------

    def fetch_initial_query_results(self, start_date: date, num_days: int) -> pd.DataFrame:
        """
        Return row counts per (deployment_hash, indexer) pair.

        Output schema: columns [deployment_hash, indexer, num_rows].
        """
        self._ensure_count_cache(start_date, num_days)

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

        Caps each (deployment_hash, indexer) pair to rows_to_use rows via
        Algorithm R reservoir sampling during pass 2. The groupby truncation
        is kept as a safety net.

        Output schema: columns [query_id, deployment_hash, fee, timestamp, blocks_behind,
                 response_time_ms, indexer, status, day_partition,
                 subgraph_network, url].
        """
        self._ensure_count_cache(start_date, num_days)
        self._ensure_row_cache(start_date, num_days, rows_to_use)

        if self._row_cache_df is None or self._row_cache_df.empty:
            logger.warning("Row cache is empty — no data in the specified Redpanda window")
            return _empty_combined_df()

        # Safety net: truncate any pair that exceeds rows_to_use after merge
        df = (
            self._row_cache_df
            .groupby(["deployment_hash", "indexer"])
            .head(rows_to_use)
            .reset_index(drop=True)
        )

        memory_mb = df.memory_usage(deep=True).sum() / (1024 * 1024)
        logger.info(f"Combined query results from Redpanda: {len(df)} rows ({memory_mb:.1f} MB)")
        return df

    def fetch_stake_to_fees(self, start_ts: str) -> pd.DataFrame:
        """
        Compute stake-to-fees ratio by combining subgraph stake data with
        fee totals accumulated during the Redpanda replay.

        stake_to_fees = (staked_tokens - locked_tokens) / total_fees_earned

        Stake comes from The Graph Network subgraph (current on-chain state).
        Fees come from self._fees_per_indexer, populated during _count_pass
        by summing fee_grt across all indexer attempts in the 28-day window.

        Returns a DataFrame indexed by 'indexer' with a 'stake_to_fees' column,
        matching the schema expected by compute_all_scores.
        """
        if not self.graph_network_url:
            logger.warning(
                "GRAPH_NETWORK_SUBGRAPH_URL not set — returning empty stake data. "
                "stake_to_fees will be NaN for all indexers."
            )
            return pd.DataFrame(columns=["stake_to_fees"])

        logger.info(f"Fetching stake data from {self.graph_network_url}")
        try:
            indexers = self._paginate_graphql_indexers()
        except Exception:
            logger.exception("Failed to fetch stake data from subgraph")
            return pd.DataFrame(columns=["stake_to_fees"])

        if not indexers:
            logger.warning("Subgraph returned no indexer records")
            return pd.DataFrame(columns=["stake_to_fees"])

        df = pd.DataFrame(indexers)
        df = df.rename(columns={"id": "indexer"})
        df["last_known_slashable_stake"] = (
            df["stakedTokens"].astype(float) - df["lockedTokens"].astype(float)
        )

        # Map fee totals from the Redpanda replay onto subgraph indexers.
        df["total_query_fees"] = df["indexer"].map(self._fees_per_indexer).fillna(0.0)

        # stake_to_fees = slashable_stake / total_fees. Indexers with zero fees
        # get NaN (division by zero produces NULL / NaN).
        df["stake_to_fees"] = (
            df["last_known_slashable_stake"] / df["total_query_fees"].replace(0.0, float("nan"))
        )

        df = df[["indexer", "stake_to_fees", "total_query_fees", "last_known_slashable_stake"]]
        df.set_index("indexer", inplace=True)

        matched = df["stake_to_fees"].notna().sum()
        logger.info(
            f"Computed stake-to-fees for {len(df)} indexers "
            f"({matched} with fee data from replay)"
        )
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

        scores_df.to_json(tmp_path, orient="records", date_format="iso", date_unit="s")
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

    def _ensure_count_cache(self, start_date: date, num_days: int) -> None:
        """Trigger a count pass only if the cache doesn't cover this window."""
        if (
            self._count_cache_start_date == start_date
            and self._count_cache_num_days == num_days
        ):
            return
        self._count_pass(start_date, num_days)

    def _ensure_row_cache(self, start_date: date, num_days: int, rows_to_use: int) -> None:
        """Trigger a sample pass only if the cache doesn't cover this window/cap."""
        if (
            self._row_cache_df is not None
            and self._row_cache_start_date == start_date
            and self._row_cache_num_days == num_days
            and self._row_cache_rows_to_use == rows_to_use
        ):
            return
        self._sample_pass(start_date, num_days, rows_to_use)

    # -----------------------------------------------------------------------
    # Internal: partition resolution
    # -----------------------------------------------------------------------

    def _resolve_partitions(self, start_ts_ms: int) -> list:
        """
        Resolve timestamp-based offsets for all partitions of self._topic.

        Creates a temporary consumer for metadata and offset resolution,
        stores result in self._cached_partitions for reuse in pass 2.
        """
        from confluent_kafka import Consumer, TopicPartition

        consumer = Consumer(self._consumer_config())
        try:
            meta = consumer.list_topics(self._topic, timeout=30)
            if self._topic not in meta.topics:
                raise RuntimeError(f"Topic '{self._topic}' not found in Redpanda")

            partition_ids = list(meta.topics[self._topic].partitions.keys())
            seek_tps = [TopicPartition(self._topic, pid, start_ts_ms) for pid in partition_ids]

            resolved = consumer.offsets_for_times(seek_tps, timeout=30)
            valid = [
                TopicPartition(tp.topic, tp.partition, tp.offset)
                for tp in resolved
                if tp.offset >= 0
            ]
        finally:
            consumer.close()

        self._cached_partitions = valid
        return valid

    # -----------------------------------------------------------------------
    # Internal: Pass 1 — count
    # -----------------------------------------------------------------------

    def _count_pass(self, start_date: date, num_days: int) -> None:
        """
        Pass 1: lightweight scan counting (deployment, indexer) pairs.

        Uses extract_keys_and_fees for minimal parsing, raw byte keys,
        batch polling, and parallel partition consumption.
        """
        start_dt = datetime.combine(start_date, datetime.min.time()).replace(tzinfo=timezone.utc)
        end_dt = datetime.now(timezone.utc)
        start_ts_ms = int(start_dt.timestamp() * 1000)
        end_ts_ms = int(end_dt.timestamp() * 1000)

        logger.info(
            f"Starting count pass: topic={self._topic}, "
            f"window={start_date} to {end_dt.isoformat()}"
        )

        partitions = self._resolve_partitions(start_ts_ms)

        if not partitions:
            logger.warning(
                "No Redpanda messages found for the requested time window — "
                f"start_ts_ms={start_ts_ms}"
            )
            self._count_cache = {}
            self._fees_per_indexer = {}
            self._count_cache_start_date = start_date
            self._count_cache_num_days = num_days
            return

        # Consume partitions in parallel
        def count_partition(tp):
            from confluent_kafka import Consumer, TopicPartition as TP
            consumer = Consumer(self._consumer_config())
            try:
                consumer.assign([TP(tp.topic, tp.partition, tp.offset)])
                return self._count_partition_loop(consumer, end_ts_ms)
            finally:
                consumer.close()

        with ThreadPoolExecutor(max_workers=min(len(partitions), MAX_PARTITION_WORKERS)) as pool:
            results = list(pool.map(count_partition, partitions))

        # Merge per-partition counts and fees
        merged_counts: Dict[Tuple[bytes, bytes], int] = defaultdict(int)
        merged_fees: Dict[bytes, float] = defaultdict(float)
        total_messages = 0
        for counts, fees, msg_count in results:
            for key, val in counts.items():
                merged_counts[key] += val
            for idx_bytes, fee in fees.items():
                merged_fees[idx_bytes] += fee
            total_messages += msg_count

        # Convert byte keys to string keys — only ~N_pairs conversions
        self._count_cache = {}
        for (dep_bytes, idx_bytes), count in merged_counts.items():
            dep_str = _bytes_to_cid(dep_bytes)
            idx_str = _bytes_to_hex(idx_bytes)
            if dep_str and idx_str:
                self._count_cache[(dep_str, idx_str)] = count

        self._fees_per_indexer = {}
        for idx_bytes, fee in merged_fees.items():
            idx_str = _bytes_to_hex(idx_bytes)
            if idx_str:
                self._fees_per_indexer[idx_str] = fee

        self._count_cache_start_date = start_date
        self._count_cache_num_days = num_days

        logger.info(
            f"Count pass complete: {total_messages:,} messages, "
            f"{sum(self._count_cache.values()):,} attempts across "
            f"{len(self._count_cache)} (deployment, indexer) pairs"
        )

    def _count_partition_loop(
        self, consumer, end_ts_ms: int
    ) -> Tuple[Dict[Tuple[bytes, bytes], int], Dict[bytes, float], int]:
        """
        Consume one partition for the count pass.

        Returns (counts_by_pair, fees_by_indexer, message_count).
        """
        counts: Dict[Tuple[bytes, bytes], int] = defaultdict(int)
        fees: Dict[bytes, float] = defaultdict(float)
        total_messages = 0
        consecutive_empty = 0

        while True:
            messages = consumer.consume(num_messages=1000, timeout=30.0)

            if not messages:
                consecutive_empty += 1
                if consecutive_empty >= 3:
                    break
                continue

            consecutive_empty = 0

            for msg in messages:
                if msg.error():
                    continue

                ts_type, ts_ms = msg.timestamp()
                if ts_ms < 0:
                    ts_ms = int(datetime.now(timezone.utc).timestamp() * 1000)

                if ts_ms > end_ts_ms:
                    return (counts, fees, total_messages)

                total_messages += 1

                try:
                    query = ClientQueryProtobuf()
                    query.ParseFromString(msg.value())
                except Exception:
                    continue

                for attempt in query.indexer_queries:
                    idx_bytes = bytes(attempt.indexer)
                    dep_bytes = bytes(attempt.deployment)
                    if len(dep_bytes) == 32 and len(idx_bytes) == 20:
                        counts[(dep_bytes, idx_bytes)] += 1
                        fees[idx_bytes] += attempt.fee_grt

        return (counts, fees, total_messages)

    # -----------------------------------------------------------------------
    # Internal: Pass 2 — sample
    # -----------------------------------------------------------------------

    def _sample_pass(self, start_date: date, num_days: int, rows_to_use: int) -> None:
        """
        Pass 2: reservoir sampling with cap = rows_to_use.

        Uses extract_sample_fields for selective parsing, raw byte keys,
        batch polling, cached partitions, deferred string conversion,
        parallel partition consumption, and thread-local PRNG.
        """
        start_dt = datetime.combine(start_date, datetime.min.time()).replace(tzinfo=timezone.utc)
        end_dt = datetime.now(timezone.utc)
        start_ts_ms = int(start_dt.timestamp() * 1000)
        end_ts_ms = int(end_dt.timestamp() * 1000)

        logger.info(
            f"Starting sample pass: topic={self._topic}, "
            f"window={start_date} to {end_dt.isoformat()}, "
            f"rows_to_use={rows_to_use}"
        )

        # Reuse cached partitions from pass 1 if available
        partitions = self._cached_partitions
        if partitions is None:
            partitions = self._resolve_partitions(start_ts_ms)

        if not partitions:
            logger.warning("No valid partitions for sample pass")
            self._row_cache_df = _empty_combined_df()
            self._row_cache_start_date = start_date
            self._row_cache_num_days = num_days
            self._row_cache_rows_to_use = rows_to_use
            return

        # Consume partitions in parallel, each with a thread-local PRNG
        def sample_partition(tp):
            from confluent_kafka import Consumer, TopicPartition as TP
            rng = random.Random()
            consumer = Consumer(self._consumer_config())
            try:
                consumer.assign([TP(tp.topic, tp.partition, tp.offset)])
                return self._sample_partition_loop(consumer, end_ts_ms, rows_to_use, rng)
            finally:
                consumer.close()

        with ThreadPoolExecutor(max_workers=min(len(partitions), MAX_PARTITION_WORKERS)) as pool:
            results = list(pool.map(sample_partition, partitions))

        # Merge per-partition reservoirs
        merged_reservoirs: Dict[Tuple[bytes, bytes], List[dict]] = defaultdict(list)
        merged_counts: Dict[Tuple[bytes, bytes], int] = defaultdict(int)
        for reservoirs, counts in results:
            for key, rows in reservoirs.items():
                merged_reservoirs[key].extend(rows)
            for key, count in counts.items():
                merged_counts[key] += count

        # Truncate pairs that exceed rows_to_use after cross-partition merge
        all_rows: List[dict] = []
        for key, rows in merged_reservoirs.items():
            if len(rows) > rows_to_use:
                random.shuffle(rows)
                rows = rows[:rows_to_use]
            all_rows.extend(rows)

        if all_rows:
            self._row_cache_df = _build_dataframe(all_rows)
        else:
            self._row_cache_df = _empty_combined_df()

        self._row_cache_start_date = start_date
        self._row_cache_num_days = num_days
        self._row_cache_rows_to_use = rows_to_use

        logger.info(
            f"Sample pass complete: {len(all_rows):,} rows buffered across "
            f"{len(merged_reservoirs)} pairs"
        )

    def _sample_partition_loop(
        self,
        consumer,
        end_ts_ms: int,
        rows_to_use: int,
        rng: random.Random,
    ) -> Tuple[Dict[Tuple[bytes, bytes], List[dict]], Dict[Tuple[bytes, bytes], int]]:
        """
        Consume one partition for the sample pass.

        Returns (reservoirs, counts) where reservoirs maps (dep_bytes, idx_bytes)
        to a list of row dicts with raw bytes and ts_ms (deferred conversion).
        """
        reservoirs: Dict[Tuple[bytes, bytes], List[dict]] = defaultdict(list)
        counts: Dict[Tuple[bytes, bytes], int] = defaultdict(int)
        consecutive_empty = 0

        while True:
            messages = consumer.consume(num_messages=1000, timeout=30.0)

            if not messages:
                consecutive_empty += 1
                if consecutive_empty >= 3:
                    break
                continue

            consecutive_empty = 0

            for msg in messages:
                if msg.error():
                    continue

                ts_type, ts_ms = msg.timestamp()
                if ts_ms < 0:
                    ts_ms = int(datetime.now(timezone.utc).timestamp() * 1000)

                if ts_ms > end_ts_ms:
                    return (reservoirs, counts)

                try:
                    query = ClientQueryProtobuf()
                    query.ParseFromString(msg.value())
                except Exception:
                    continue

                for attempt in query.indexer_queries:
                    idx_bytes = bytes(attempt.indexer)
                    dep_bytes = bytes(attempt.deployment)
                    url = attempt.url

                    if len(dep_bytes) != 32 or len(idx_bytes) != 20 or not url:
                        continue

                    key = (dep_bytes, idx_bytes)
                    n = counts[key]

                    row = {
                        "query_id": query.query_id,
                        "indexer_bytes": idx_bytes,
                        "deployment_bytes": dep_bytes,
                        "fee": attempt.fee_grt,
                        "ts_ms": ts_ms,
                        "blocks_behind": attempt.blocks_behind,
                        "response_time_ms": attempt.response_time_ms,
                        "status": _map_result_to_status(attempt.result),
                        "subgraph_network": attempt.indexed_chain,
                        "url": url if url.endswith("/") else url + "/",
                    }

                    if n < rows_to_use:
                        reservoirs[key].append(row)
                    else:
                        j = rng.randint(0, n)
                        if j < rows_to_use:
                            reservoirs[key][j] = row

                    counts[key] += 1

        return (reservoirs, counts)

    # -----------------------------------------------------------------------
    # Internal: GraphQL pagination
    # -----------------------------------------------------------------------

    @retry(
        retry=retry_if_exception_type((
            ConnectionError,
            requests.exceptions.ConnectionError,
            requests.exceptions.Timeout,
            requests.exceptions.HTTPError,
        )),
        stop=stop_after_attempt(5),
        wait=wait_exponential(multiplier=2, max=60),
        reraise=True,
    )
    def _paginate_graphql_indexers(self) -> List[dict]:
        """Paginate through all indexers from the Graph Network subgraph.

        Uses cursor-based pagination (id_gt) instead of skip-based to avoid
        progressively slower queries as the offset grows.
        """
        query = """
        query($first: Int!, $lastId: String!) {
          indexers(first: $first, where: { id_gt: $lastId }, orderBy: id) {
            id
            stakedTokens
            lockedTokens
          }
        }
        """
        return paginate_subgraph_query(self.graph_network_url, query, entity="indexers")


# ---------------------------------------------------------------------------
# Helpers
# ---------------------------------------------------------------------------


def _build_dataframe(rows: List[dict]) -> pd.DataFrame:
    """
    Convert raw row dicts (with byte keys and ts_ms) into the final DataFrame.

    Performs deferred string conversions in bulk rather than per-message
    during streaming.
    """
    df = pd.DataFrame(rows)

    dep_map = {b: _bytes_to_cid(b) for b in df["deployment_bytes"].unique()}
    idx_map = {b: _bytes_to_hex(b) for b in df["indexer_bytes"].unique()}
    df["deployment_hash"] = df["deployment_bytes"].map(dep_map)
    df["indexer"] = df["indexer_bytes"].map(idx_map)
    df["timestamp"] = pd.to_datetime(df["ts_ms"], unit="ms", utc=True)
    df["day_partition"] = df["timestamp"].dt.date

    df = df.drop(columns=["deployment_bytes", "indexer_bytes", "ts_ms"])
    return df


def _empty_combined_df() -> pd.DataFrame:
    return pd.DataFrame(
        columns=[
            "query_id", "deployment_hash", "fee", "timestamp",
            "blocks_behind", "response_time_ms", "indexer", "status",
            "day_partition", "subgraph_network", "url",
        ]
    )
