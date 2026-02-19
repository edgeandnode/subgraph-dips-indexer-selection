"""
Unit tests for RedpandaProvider and its supporting utilities.

Tests use the path-insertion pattern established in test_compute_scores_job.py
so that imports resolve correctly without installing the cronjob package.
"""

import json
import os
import struct
import sys
import tempfile
from collections import namedtuple
from datetime import date, datetime, timezone
from pathlib import Path
from typing import Dict, List
from unittest.mock import MagicMock, patch, call

import pandas as pd
import pytest

# Make the cronjob package importable.
jobs_path = Path(__file__).parent.parent / "cronjobs" / "compute_scores"
sys.path.insert(0, str(jobs_path))

import importlib.util
import types

# The installed proto-plus package claims the `proto` namespace, shadowing the
# local cronjobs/compute_scores/proto/ directory. Pre-register the local proto
# package and its pb2 module in sys.modules so that both this test file and
# redpanda.py resolve `from proto.gateway_queries_pb2 import ...` correctly.
_local_proto_pkg = types.ModuleType("proto")
_local_proto_pkg.__path__ = [str(jobs_path / "proto")]
_local_proto_pkg.__package__ = "proto"
sys.modules["proto"] = _local_proto_pkg

_pb2_spec = importlib.util.spec_from_file_location(
    "proto.gateway_queries_pb2",
    jobs_path / "proto" / "gateway_queries_pb2.py",
)
_pb2_mod = importlib.util.module_from_spec(_pb2_spec)
_pb2_spec.loader.exec_module(_pb2_mod)
sys.modules["proto.gateway_queries_pb2"] = _pb2_mod

ClientQueryProtobuf = _pb2_mod.ClientQueryProtobuf
IndexerQueryProtobuf = _pb2_mod.IndexerQueryProtobuf
extract_keys_and_fees = _pb2_mod.extract_keys_and_fees
extract_sample_fields = _pb2_mod.extract_sample_fields

from redpanda import (
    RedpandaProvider,
    _bytes_to_cid,
    _bytes_to_hex,
    _map_result_to_status,
)


# ---------------------------------------------------------------------------
# Proto binary helpers — encode simple protobuf messages for testing
# ---------------------------------------------------------------------------


def _encode_varint(value: int) -> bytes:
    """Encode an integer as a protobuf varint."""
    buf = bytearray()
    while True:
        bits = value & 0x7F
        value >>= 7
        if value:
            buf.append(bits | 0x80)
        else:
            buf.append(bits)
            break
    return bytes(buf)


def _encode_field_varint(field_number: int, value: int) -> bytes:
    tag = (field_number << 3) | 0  # wire type 0
    return _encode_varint(tag) + _encode_varint(value)


def _encode_field_double(field_number: int, value: float) -> bytes:
    tag = (field_number << 3) | 1  # wire type 1
    return _encode_varint(tag) + struct.pack("<d", value)


def _encode_field_bytes(field_number: int, data: bytes) -> bytes:
    tag = (field_number << 3) | 2  # wire type 2
    return _encode_varint(tag) + _encode_varint(len(data)) + data


def _encode_field_str(field_number: int, text: str) -> bytes:
    return _encode_field_bytes(field_number, text.encode("utf-8"))


def _build_indexer_attempt(
    indexer: bytes,
    deployment: bytes,
    indexed_chain: str,
    url: str,
    fee_grt: float,
    response_time_ms: int,
    result: str,
    blocks_behind: int,
) -> bytes:
    """Encode an IndexerQueryProtobuf message."""
    buf = b""
    buf += _encode_field_bytes(1, indexer)       # indexer
    buf += _encode_field_bytes(2, deployment)    # deployment
    buf += _encode_field_bytes(3, b"\x00" * 20) # allocation (unused)
    buf += _encode_field_str(4, indexed_chain)  # indexed_chain
    buf += _encode_field_str(5, url)             # url
    buf += _encode_field_double(6, fee_grt)      # fee_grt
    buf += _encode_field_varint(7, response_time_ms)  # response_time_ms
    buf += _encode_field_str(9, result)          # result
    buf += _encode_field_varint(11, blocks_behind)    # blocks_behind
    return buf


def _build_client_query(
    query_id: str,
    attempts: List[bytes],
) -> bytes:
    """Encode a ClientQueryProtobuf message."""
    buf = b""
    buf += _encode_field_str(3, query_id)  # query_id
    for attempt_bytes in attempts:
        buf += _encode_field_bytes(10, attempt_bytes)  # indexer_queries
    return buf


# ---------------------------------------------------------------------------
# Reference data
# ---------------------------------------------------------------------------

INDEXER_BYTES = bytes.fromhex("abcdef1234567890abcdef1234567890abcdef12")
DEPLOYMENT_BYTES = bytes(range(32))  # 32 distinct bytes
EXPECTED_INDEXER = "0x" + INDEXER_BYTES.hex().lower()
EXPECTED_DEPLOYMENT_CID = _bytes_to_cid(DEPLOYMENT_BYTES)


# ---------------------------------------------------------------------------
# Encoding helper tests
# ---------------------------------------------------------------------------


class TestBytesToCid:
    def test_valid_32_bytes(self):
        # Arrange
        b = bytes(range(32))

        # Act
        result = _bytes_to_cid(b)

        # Assert
        assert result.startswith("Qm"), f"Expected CIDv0 starting with Qm, got {result!r}"
        assert len(result) == 46, f"Expected CIDv0 length 46, got {len(result)}"

    def test_wrong_length_returns_empty(self):
        assert _bytes_to_cid(b"\x01\x02\x03") == ""
        assert _bytes_to_cid(b"") == ""

    def test_deterministic(self):
        b = bytes(range(32))
        assert _bytes_to_cid(b) == _bytes_to_cid(b)


class TestBytesToHex:
    def test_valid_20_bytes(self):
        # Arrange
        b = bytes.fromhex("abcdef1234567890abcdef1234567890abcdef12")

        # Act
        result = _bytes_to_hex(b)

        # Assert
        assert result == "0xabcdef1234567890abcdef1234567890abcdef12"

    def test_wrong_length_returns_empty(self):
        assert _bytes_to_hex(b"\x01") == ""
        assert _bytes_to_hex(b"") == ""

    def test_lowercase(self):
        b = bytes.fromhex("ABCDEF1234567890ABCDEF1234567890ABCDEF12")
        assert _bytes_to_hex(b) == "0x" + "abcdef1234567890abcdef1234567890abcdef12"


class TestMapResultToStatus:
    def test_success_maps_to_200_ok(self):
        assert _map_result_to_status("success") == "200 OK"

    def test_other_values_pass_through(self):
        assert _map_result_to_status("Unavailable(MissingBlock)") == "Unavailable(MissingBlock)"
        assert _map_result_to_status("Timeout") == "Timeout"
        assert _map_result_to_status("") == ""


# ---------------------------------------------------------------------------
# Proto parser tests (existing class-based)
# ---------------------------------------------------------------------------


class TestIndexerQueryProtobufParsing:
    def test_round_trip(self):
        # Arrange
        raw = _build_indexer_attempt(
            indexer=INDEXER_BYTES,
            deployment=DEPLOYMENT_BYTES,
            indexed_chain="mainnet",
            url="https://indexer.example.com/",
            fee_grt=0.001,
            response_time_ms=42,
            result="success",
            blocks_behind=0,
        )

        # Act
        msg = IndexerQueryProtobuf.FromString(raw)

        # Assert
        assert msg.indexer == INDEXER_BYTES
        assert msg.deployment == DEPLOYMENT_BYTES
        assert msg.indexed_chain == "mainnet"
        assert msg.url == "https://indexer.example.com/"
        assert abs(msg.fee_grt - 0.001) < 1e-9
        assert msg.response_time_ms == 42
        assert msg.result == "success"
        assert msg.blocks_behind == 0

    def test_empty_bytes_parse_without_error(self):
        msg = IndexerQueryProtobuf.FromString(b"")
        assert msg.indexer == b""
        assert msg.result == ""


class TestClientQueryProtobufParsing:
    def test_round_trip_with_attempts(self):
        # Arrange
        attempt_raw = _build_indexer_attempt(
            indexer=INDEXER_BYTES,
            deployment=DEPLOYMENT_BYTES,
            indexed_chain="mainnet",
            url="https://indexer.example.com/",
            fee_grt=0.001,
            response_time_ms=100,
            result="success",
            blocks_behind=5,
        )
        raw = _build_client_query(
            query_id="abc123def456-JFK",
            attempts=[attempt_raw],
        )

        # Act
        msg = ClientQueryProtobuf.FromString(raw)

        # Assert
        assert msg.query_id == "abc123def456-JFK"
        assert len(msg.indexer_queries) == 1
        attempt = msg.indexer_queries[0]
        assert attempt.result == "success"
        assert attempt.response_time_ms == 100
        assert attempt.blocks_behind == 5

    def test_multiple_attempts(self):
        # Arrange
        attempts = [
            _build_indexer_attempt(
                indexer=INDEXER_BYTES,
                deployment=DEPLOYMENT_BYTES,
                indexed_chain="mainnet",
                url="https://a.example.com/",
                fee_grt=0.001,
                response_time_ms=100,
                result="success",
                blocks_behind=0,
            ),
            _build_indexer_attempt(
                indexer=bytes(20),  # second indexer
                deployment=DEPLOYMENT_BYTES,
                indexed_chain="mainnet",
                url="https://b.example.com/",
                fee_grt=0.002,
                response_time_ms=200,
                result="Timeout",
                blocks_behind=10,
            ),
        ]
        raw = _build_client_query("qid-001-LAX", attempts)

        # Act
        msg = ClientQueryProtobuf.FromString(raw)

        # Assert
        assert len(msg.indexer_queries) == 2
        assert msg.indexer_queries[0].result == "success"
        assert msg.indexer_queries[1].result == "Timeout"


# ---------------------------------------------------------------------------
# Proto extraction function tests (optimized parsers)
# ---------------------------------------------------------------------------


class TestExtractKeysAndFees:
    """Tests for the pass 1 minimal parser."""

    def test_returns_correct_tuples(self):
        # Arrange — two-attempt message
        a1 = _build_indexer_attempt(
            indexer=INDEXER_BYTES, deployment=DEPLOYMENT_BYTES,
            indexed_chain="mainnet", url="https://a.example.com/",
            fee_grt=0.001, response_time_ms=50, result="success", blocks_behind=0,
        )
        a2 = _build_indexer_attempt(
            indexer=bytes(20), deployment=DEPLOYMENT_BYTES,
            indexed_chain="mainnet", url="https://b.example.com/",
            fee_grt=0.002, response_time_ms=80, result="Timeout", blocks_behind=3,
        )
        raw = _build_client_query("qid-001-JFK", [a1, a2])

        # Act
        results = extract_keys_and_fees(raw)

        # Assert
        assert len(results) == 2
        idx1, dep1, fee1 = results[0]
        assert idx1 == INDEXER_BYTES
        assert dep1 == DEPLOYMENT_BYTES
        assert abs(fee1 - 0.001) < 1e-9

        idx2, dep2, fee2 = results[1]
        assert idx2 == bytes(20)
        assert dep2 == DEPLOYMENT_BYTES
        assert abs(fee2 - 0.002) < 1e-9

    def test_empty_message_returns_empty_list(self):
        # A message with no indexer_queries field
        raw = _encode_field_str(3, "qid-empty")

        # Act
        results = extract_keys_and_fees(raw)

        # Assert
        assert results == []

    def test_empty_bytes_returns_empty_list(self):
        assert extract_keys_and_fees(b"") == []


class TestExtractSampleFields:
    """Tests for the pass 2 selective parser."""

    def test_returns_query_id_and_attempts(self):
        # Arrange
        a1 = _build_indexer_attempt(
            indexer=INDEXER_BYTES, deployment=DEPLOYMENT_BYTES,
            indexed_chain="mainnet", url="https://indexer.example.com/",
            fee_grt=0.001, response_time_ms=42, result="success", blocks_behind=5,
        )
        raw = _build_client_query("abc123-JFK", [a1])

        # Act
        query_id, attempts = extract_sample_fields(raw)

        # Assert
        assert query_id == "abc123-JFK"
        assert len(attempts) == 1

        att = attempts[0]
        assert att["indexer_bytes"] == INDEXER_BYTES
        assert att["deployment_bytes"] == DEPLOYMENT_BYTES
        assert att["indexed_chain"] == "mainnet"
        assert att["url"] == "https://indexer.example.com/"
        assert abs(att["fee_grt"] - 0.001) < 1e-9
        assert att["response_time_ms"] == 42
        assert att["result"] == "success"
        assert att["blocks_behind"] == 5

    def test_skipped_fields_not_present(self):
        """allocation (field 3), seconds_behind (8), indexer_errors (10) are skipped."""
        a1 = _build_indexer_attempt(
            indexer=INDEXER_BYTES, deployment=DEPLOYMENT_BYTES,
            indexed_chain="mainnet", url="https://indexer.example.com/",
            fee_grt=0.001, response_time_ms=42, result="success", blocks_behind=0,
        )
        raw = _build_client_query("qid-skip", [a1])

        query_id, attempts = extract_sample_fields(raw)
        att = attempts[0]

        # These keys should NOT be in the dict
        assert "allocation" not in att
        assert "seconds_behind" not in att
        assert "indexer_errors" not in att

    def test_multiple_attempts(self):
        a1 = _build_indexer_attempt(
            indexer=INDEXER_BYTES, deployment=DEPLOYMENT_BYTES,
            indexed_chain="mainnet", url="https://a.example.com/",
            fee_grt=0.001, response_time_ms=50, result="success", blocks_behind=0,
        )
        a2 = _build_indexer_attempt(
            indexer=bytes(20), deployment=DEPLOYMENT_BYTES,
            indexed_chain="arbitrum-one", url="https://b.example.com/",
            fee_grt=0.002, response_time_ms=80, result="Timeout", blocks_behind=10,
        )
        raw = _build_client_query("qid-multi", [a1, a2])

        query_id, attempts = extract_sample_fields(raw)

        assert query_id == "qid-multi"
        assert len(attempts) == 2
        assert attempts[0]["indexed_chain"] == "mainnet"
        assert attempts[1]["indexed_chain"] == "arbitrum-one"
        assert attempts[1]["result"] == "Timeout"

    def test_empty_bytes(self):
        query_id, attempts = extract_sample_fields(b"")
        assert query_id == ""
        assert attempts == []


# ---------------------------------------------------------------------------
# RedpandaProvider: write_scores / scores_exist_for_today
# ---------------------------------------------------------------------------


class TestScoresFileIO:
    def _make_scores_df(self, today: bool = True) -> pd.DataFrame:
        """Build a minimal scores DataFrame."""
        computed_at = datetime.now(timezone.utc) if today else datetime(2000, 1, 1, tzinfo=timezone.utc)
        return pd.DataFrame(
            [
                {
                    "indexer": EXPECTED_INDEXER,
                    "url": "https://indexer.example.com/",
                    "lat_normalized_score": 0.8,
                    "uptime_score": 0.95,
                    "success_rate": 0.99,
                    "stake_to_fees": float("nan"),
                    "computed_at": computed_at,
                }
            ]
        )

    def test_write_scores_creates_file(self, tmp_path):
        # Arrange
        scores_path = str(tmp_path / "scores" / "indexer_scores.json")
        provider = RedpandaProvider.__new__(RedpandaProvider)
        provider._scores_path = scores_path

        # Act
        provider.write_scores(self._make_scores_df())

        # Assert
        assert Path(scores_path).exists()
        with open(scores_path) as f:
            data = json.load(f)
        assert len(data) == 1
        assert data[0]["indexer"] == EXPECTED_INDEXER

    def test_scores_exist_for_today_true(self, tmp_path):
        # Arrange
        scores_path = str(tmp_path / "indexer_scores.json")
        provider = RedpandaProvider.__new__(RedpandaProvider)
        provider._scores_path = scores_path
        provider.write_scores(self._make_scores_df(today=True))

        # Act / Assert
        assert provider.scores_exist_for_today() is True

    def test_scores_exist_for_today_false_old_file(self, tmp_path):
        # Arrange
        scores_path = str(tmp_path / "indexer_scores.json")
        provider = RedpandaProvider.__new__(RedpandaProvider)
        provider._scores_path = scores_path
        provider.write_scores(self._make_scores_df(today=False))

        # Act / Assert
        assert provider.scores_exist_for_today() is False

    def test_scores_exist_for_today_false_missing_file(self, tmp_path):
        # Arrange
        provider = RedpandaProvider.__new__(RedpandaProvider)
        provider._scores_path = str(tmp_path / "nonexistent.json")

        # Act / Assert
        assert provider.scores_exist_for_today() is False

    def test_write_scores_atomic(self, tmp_path):
        """write_scores must not leave a .tmp file on disk."""
        scores_path = str(tmp_path / "indexer_scores.json")
        provider = RedpandaProvider.__new__(RedpandaProvider)
        provider._scores_path = scores_path
        provider.write_scores(self._make_scores_df())

        assert not Path(scores_path + ".tmp").exists()
        assert Path(scores_path).exists()


# ---------------------------------------------------------------------------
# Mock helpers for two-pass architecture
# ---------------------------------------------------------------------------


def _fake_kafka_message(ts_ms: int, value: bytes, partition: int = 0, offset: int = 0):
    """Create a minimal mock confluent_kafka Message."""
    msg = MagicMock()
    msg.error.return_value = None
    msg.timestamp.return_value = (1, ts_ms)  # TIMESTAMP_CREATE_TIME
    msg.value.return_value = value
    msg.partition.return_value = partition
    msg.offset.return_value = offset
    return msg


def _make_mock_consumer_class(consumer_mocks: List[MagicMock]) -> MagicMock:
    """
    Build a mock Consumer class that returns a different mock instance
    on each instantiation (resolution consumer, then per-partition consumers).
    """
    MockClass = MagicMock()
    MockClass.side_effect = consumer_mocks
    return MockClass


def _make_resolution_consumer(topic: str, partition_ids: List[int], start_offset: int = 100):
    """Build a mock consumer for partition resolution (list_topics + offsets_for_times)."""
    mock = MagicMock()

    mock_topic_meta = MagicMock()
    mock_topic_meta.partitions = {pid: MagicMock() for pid in partition_ids}
    mock.list_topics.return_value.topics = {topic: mock_topic_meta}

    resolved_tps = []
    for pid in partition_ids:
        tp = MagicMock()
        tp.topic = topic
        tp.partition = pid
        tp.offset = start_offset
        resolved_tps.append(tp)
    mock.offsets_for_times.return_value = resolved_tps

    return mock


def _make_partition_consumer(message_batches: List[List]):
    """
    Build a mock consumer for a single partition.

    message_batches is a list of lists: each inner list is returned by
    one call to consume(). Empty lists simulate timeouts.
    """
    mock = MagicMock()
    mock.consume.side_effect = message_batches
    return mock


# ---------------------------------------------------------------------------
# RedpandaProvider: two-pass caching tests
# ---------------------------------------------------------------------------


class TestRedpandaProviderCaching:
    """Tests for _count_pass, _sample_pass, fetch_initial_query_results, fetch_combined_query_results."""

    def _build_provider(self) -> RedpandaProvider:
        """Build a RedpandaProvider with mocked environment."""
        with patch.dict(
            os.environ,
            {
                "REDPANDA_BOOTSTRAP_SERVERS": "localhost:9092",
                "REDPANDA_TOPIC": "gateway_queries",
            },
        ):
            return RedpandaProvider()

    def _two_attempt_message(
        self,
        query_id: str,
        deployment: bytes,
        indexer1: bytes,
        indexer2: bytes,
        ts_ms: int,
    ) -> bytes:
        """Encode a ClientQueryProtobuf with two indexer attempts."""
        a1 = _build_indexer_attempt(
            indexer=indexer1,
            deployment=deployment,
            indexed_chain="mainnet",
            url="https://indexer1.example.com/",
            fee_grt=0.001,
            response_time_ms=50,
            result="success",
            blocks_behind=0,
        )
        a2 = _build_indexer_attempt(
            indexer=indexer2,
            deployment=deployment,
            indexed_chain="mainnet",
            url="https://indexer2.example.com/",
            fee_grt=0.002,
            response_time_ms=80,
            result="Unavailable(MissingBlock)",
            blocks_behind=3,
        )
        return _build_client_query(query_id, [a1, a2])

    def test_fetch_initial_query_results_counts(self):
        """fetch_initial_query_results should return true attempt counts (count pass only)."""
        # Arrange
        provider = self._build_provider()
        start_date = date(2024, 1, 1)
        num_days = 1
        ts_ms = int(datetime(2024, 1, 1, 12, 0, tzinfo=timezone.utc).timestamp() * 1000)

        deployment = bytes(range(32))
        indexer1 = INDEXER_BYTES
        indexer2 = bytes(20)

        msg_value = self._two_attempt_message(
            "qid-001-JFK", deployment, indexer1, indexer2, ts_ms
        )
        fake_msg = _fake_kafka_message(ts_ms, msg_value)

        # Resolution consumer (for _resolve_partitions)
        resolution_consumer = _make_resolution_consumer("gateway_queries", [0])

        # Count pass partition consumer — one batch of messages then 3 empty batches
        count_consumer = _make_partition_consumer([
            [fake_msg],  # batch 1: one message
            [], [], [],  # 3 empty = end of data
        ])

        with patch("confluent_kafka.Consumer", side_effect=[resolution_consumer, count_consumer]):
            result = provider.fetch_initial_query_results(start_date, num_days)

        # Assert — two (deployment, indexer) pairs, each seen once
        assert len(result) == 2, f"Expected 2 pairs, got {len(result)}: {result}"
        assert set(result["num_rows"].tolist()) == {1}

    def test_fetch_combined_query_results_rows_to_use_cap(self):
        """fetch_combined_query_results should cap each pair at rows_to_use."""
        # Arrange
        provider = self._build_provider()
        start_date = date(2024, 1, 1)
        num_days = 1
        base_ts_ms = int(datetime(2024, 1, 1, 12, 0, tzinfo=timezone.utc).timestamp() * 1000)

        deployment = bytes(range(32))
        indexer1 = INDEXER_BYTES

        # Build 5 messages for the same (deployment, indexer1) pair
        messages = []
        for i in range(5):
            a = _build_indexer_attempt(
                indexer=indexer1,
                deployment=deployment,
                indexed_chain="mainnet",
                url="https://indexer1.example.com/",
                fee_grt=0.001,
                response_time_ms=50 + i,
                result="success",
                blocks_behind=0,
            )
            raw = _build_client_query(f"qid-{i:03d}-JFK", [a])
            messages.append(_fake_kafka_message(base_ts_ms + i * 1000, raw))

        # Resolution consumer (shared for both passes)
        resolution_consumer = _make_resolution_consumer("gateway_queries", [0])

        # Count pass consumer
        count_consumer = _make_partition_consumer([
            messages,   # all 5 messages in one batch
            [], [], [],
        ])

        # Sample pass consumer (same messages replayed)
        sample_consumer = _make_partition_consumer([
            messages,
            [], [], [],
        ])

        with patch(
            "confluent_kafka.Consumer",
            side_effect=[resolution_consumer, count_consumer, sample_consumer],
        ):
            result = provider.fetch_combined_query_results(start_date, num_days, rows_to_use=3)

        # Assert — capped at 3
        assert len(result) == 3, f"Expected 3 rows (cap), got {len(result)}"

    def test_second_call_uses_count_cache(self):
        """The second fetch_initial call must not trigger a second count pass."""
        provider = self._build_provider()
        start_date = date(2024, 1, 1)
        num_days = 1

        # Seed the count cache directly
        provider._count_cache_start_date = start_date
        provider._count_cache_num_days = num_days
        provider._count_cache = {(_bytes_to_cid(bytes(32)), _bytes_to_hex(INDEXER_BYTES)): 10}
        provider._fees_per_indexer = {_bytes_to_hex(INDEXER_BYTES): 0.01}

        with patch.object(provider, "_count_pass") as mock_count:
            result = provider.fetch_initial_query_results(start_date, num_days)

        mock_count.assert_not_called()
        assert len(result) == 1

    def test_second_call_uses_row_cache(self):
        """The second fetch_combined call must not trigger a second sample pass."""
        provider = self._build_provider()
        start_date = date(2024, 1, 1)
        num_days = 1
        rows_to_use = 100

        # Seed both caches
        provider._count_cache_start_date = start_date
        provider._count_cache_num_days = num_days
        provider._count_cache = {(_bytes_to_cid(bytes(32)), _bytes_to_hex(INDEXER_BYTES)): 10}
        provider._fees_per_indexer = {}

        provider._row_cache_start_date = start_date
        provider._row_cache_num_days = num_days
        provider._row_cache_rows_to_use = rows_to_use
        provider._row_cache_df = pd.DataFrame(
            [{"query_id": "x", "deployment_hash": _bytes_to_cid(bytes(32)),
              "fee": 0.001, "timestamp": pd.Timestamp("2024-01-01", tz="UTC"),
              "blocks_behind": 0, "response_time_ms": 100,
              "indexer": _bytes_to_hex(INDEXER_BYTES), "status": "200 OK",
              "day_partition": date(2024, 1, 1), "subgraph_network": "mainnet",
              "url": "https://indexer.example.com/"}]
        )

        with patch.object(provider, "_count_pass") as mock_count, \
             patch.object(provider, "_sample_pass") as mock_sample:
            result = provider.fetch_combined_query_results(start_date, num_days, rows_to_use)

        mock_count.assert_not_called()
        mock_sample.assert_not_called()
        assert len(result) == 1

    def test_empty_partitions_produce_empty_caches(self):
        """When no valid partitions exist, caches should be empty."""
        provider = self._build_provider()
        start_date = date(2024, 1, 1)
        num_days = 1

        # Resolution consumer returns no valid offsets (offset = -1001)
        resolution_consumer = MagicMock()
        mock_topic_meta = MagicMock()
        mock_topic_meta.partitions = {0: MagicMock()}
        resolution_consumer.list_topics.return_value.topics = {"gateway_queries": mock_topic_meta}

        invalid_tp = MagicMock()
        invalid_tp.topic = "gateway_queries"
        invalid_tp.partition = 0
        invalid_tp.offset = -1001  # OFFSET_INVALID
        resolution_consumer.offsets_for_times.return_value = [invalid_tp]

        with patch("confluent_kafka.Consumer", side_effect=[resolution_consumer]):
            result = provider.fetch_initial_query_results(start_date, num_days)

        assert result.empty
        assert provider._count_cache == {}
        assert provider._fees_per_indexer == {}


# ---------------------------------------------------------------------------
# Parallel merge test
# ---------------------------------------------------------------------------


class TestParallelMerge:
    """Verify cross-partition merge correctness."""

    def _build_provider(self) -> RedpandaProvider:
        with patch.dict(
            os.environ,
            {
                "REDPANDA_BOOTSTRAP_SERVERS": "localhost:9092",
                "REDPANDA_TOPIC": "gateway_queries",
            },
        ):
            return RedpandaProvider()

    def test_two_partitions_merge_counts_and_cap(self):
        """
        Two partitions with overlapping pairs: merged counts sum correctly
        and merged reservoirs don't exceed rows_to_use.
        """
        provider = self._build_provider()
        start_date = date(2024, 1, 1)
        num_days = 1
        base_ts_ms = int(datetime(2024, 1, 1, 12, 0, tzinfo=timezone.utc).timestamp() * 1000)

        deployment = bytes(range(32))
        indexer1 = INDEXER_BYTES

        # 4 messages per partition for the same pair, 8 total
        def make_messages(offset_start):
            msgs = []
            for i in range(4):
                a = _build_indexer_attempt(
                    indexer=indexer1, deployment=deployment,
                    indexed_chain="mainnet", url="https://indexer1.example.com/",
                    fee_grt=0.001, response_time_ms=50 + i,
                    result="success", blocks_behind=0,
                )
                raw = _build_client_query(f"qid-{offset_start + i:03d}-JFK", [a])
                msgs.append(_fake_kafka_message(base_ts_ms + i * 1000, raw))
            return msgs

        p0_msgs = make_messages(0)
        p1_msgs = make_messages(100)

        # Resolution consumer: 2 partitions
        resolution_consumer = _make_resolution_consumer("gateway_queries", [0, 1])

        # Count pass: one consumer per partition
        count_p0 = _make_partition_consumer([p0_msgs, [], [], []])
        count_p1 = _make_partition_consumer([p1_msgs, [], [], []])

        # Sample pass: replay the same messages
        sample_p0 = _make_partition_consumer([p0_msgs, [], [], []])
        sample_p1 = _make_partition_consumer([p1_msgs, [], [], []])

        with patch(
            "confluent_kafka.Consumer",
            side_effect=[
                resolution_consumer,   # partition resolution
                count_p0, count_p1,    # count pass (2 partitions)
                sample_p0, sample_p1,  # sample pass (2 partitions)
            ],
        ):
            # Count pass
            initial = provider.fetch_initial_query_results(start_date, num_days)

            # Verify merged count = 4 + 4 = 8
            assert len(initial) == 1
            assert initial.iloc[0]["num_rows"] == 8

            # Sample pass with cap = 3
            combined = provider.fetch_combined_query_results(start_date, num_days, rows_to_use=3)

        assert len(combined) == 3, f"Expected 3 rows (cap), got {len(combined)}"


# ---------------------------------------------------------------------------
# Field mapping: status column
# ---------------------------------------------------------------------------


class TestFieldMapping:
    def test_success_result_produces_200_ok_status(self):
        # Encode a single-attempt message with result="success"
        attempt = _build_indexer_attempt(
            indexer=INDEXER_BYTES,
            deployment=DEPLOYMENT_BYTES,
            indexed_chain="mainnet",
            url="https://indexer.example.com/",
            fee_grt=0.001,
            response_time_ms=100,
            result="success",
            blocks_behind=0,
        )
        msg = IndexerQueryProtobuf.FromString(attempt)
        status = _map_result_to_status(msg.result)
        assert status == "200 OK"

    def test_non_success_result_passes_through(self):
        attempt = _build_indexer_attempt(
            indexer=INDEXER_BYTES,
            deployment=DEPLOYMENT_BYTES,
            indexed_chain="mainnet",
            url="https://indexer.example.com/",
            fee_grt=0.001,
            response_time_ms=100,
            result="Unavailable(MissingBlock)",
            blocks_behind=0,
        )
        msg = IndexerQueryProtobuf.FromString(attempt)
        status = _map_result_to_status(msg.result)
        assert status == "Unavailable(MissingBlock)"

    def test_indexer_hex_encoding(self):
        assert _bytes_to_hex(INDEXER_BYTES).startswith("0x")
        assert len(_bytes_to_hex(INDEXER_BYTES)) == 42

    def test_deployment_cid_encoding(self):
        cid = _bytes_to_cid(DEPLOYMENT_BYTES)
        assert cid.startswith("Qm")
        assert len(cid) == 46


# ---------------------------------------------------------------------------
# RedpandaProvider: fetch_stake_to_fees
# ---------------------------------------------------------------------------

INDEXER2_BYTES = bytes(20)  # 20 zero bytes
EXPECTED_INDEXER2 = _bytes_to_hex(INDEXER2_BYTES)


class TestFetchStakeToFees:
    """Tests for stake-to-fees ratio computed from subgraph stake + replay fees."""

    def _build_provider_with_fees(self, fees: Dict[str, float]) -> RedpandaProvider:
        """Build a RedpandaProvider with pre-populated fee cache."""
        with patch.dict(
            os.environ,
            {
                "REDPANDA_BOOTSTRAP_SERVERS": "localhost:9092",
                "GRAPH_NETWORK_SUBGRAPH_URL": "http://graph-node:8000/subgraphs/name/graph-network",
            },
        ):
            provider = RedpandaProvider()
        provider._fees_per_indexer = fees
        return provider

    def _mock_subgraph_response(self, indexers: List[dict]):
        """Build a mock requests.post that returns indexer data."""
        response = MagicMock()
        response.status_code = 200
        response.json.return_value = {"data": {"indexers": indexers}}
        response.raise_for_status = MagicMock()
        return response

    def test_computes_ratio_from_stake_and_fees(self):
        """stake_to_fees = (stakedTokens - lockedTokens) / total_fees."""
        # Arrange
        provider = self._build_provider_with_fees({
            EXPECTED_INDEXER: 100.0,   # earned 100 GRT in fees
            EXPECTED_INDEXER2: 50.0,   # earned 50 GRT in fees
        })

        subgraph_indexers = [
            {"id": EXPECTED_INDEXER, "stakedTokens": "1000000", "lockedTokens": "0"},
            {"id": EXPECTED_INDEXER2, "stakedTokens": "500000", "lockedTokens": "100000"},
        ]

        # Act
        with patch("redpanda.requests.post", return_value=self._mock_subgraph_response(subgraph_indexers)):
            result = provider.fetch_stake_to_fees("2024-01-01T00:00:00Z")

        # Assert
        # Indexer 1: (1000000 - 0) / 100 = 10000
        assert abs(result.loc[EXPECTED_INDEXER, "stake_to_fees"] - 10000.0) < 1e-6
        # Indexer 2: (500000 - 100000) / 50 = 8000
        assert abs(result.loc[EXPECTED_INDEXER2, "stake_to_fees"] - 8000.0) < 1e-6

    def test_zero_fees_produces_nan(self):
        """Indexers with zero fees should get NaN, not infinity."""
        # Arrange — indexer has stake but earned no fees during the window
        provider = self._build_provider_with_fees({})

        subgraph_indexers = [
            {"id": EXPECTED_INDEXER, "stakedTokens": "1000000", "lockedTokens": "0"},
        ]

        # Act
        with patch("redpanda.requests.post", return_value=self._mock_subgraph_response(subgraph_indexers)):
            result = provider.fetch_stake_to_fees("2024-01-01T00:00:00Z")

        # Assert
        assert pd.isna(result.loc[EXPECTED_INDEXER, "stake_to_fees"])

    def test_missing_subgraph_url_returns_empty(self):
        """Without GRAPH_NETWORK_SUBGRAPH_URL, return empty DataFrame."""
        # Arrange
        with patch.dict(os.environ, {"REDPANDA_BOOTSTRAP_SERVERS": "localhost:9092"}, clear=False):
            with patch.dict(os.environ, {"GRAPH_NETWORK_SUBGRAPH_URL": ""}, clear=False):
                provider = RedpandaProvider()

        # Act
        result = provider.fetch_stake_to_fees("2024-01-01T00:00:00Z")

        # Assert
        assert result.empty
        assert "stake_to_fees" in result.columns

    def test_subgraph_indexer_not_in_replay_gets_nan(self):
        """Indexers in the subgraph but absent from the replay get NaN."""
        # Arrange — fee cache has no data for this indexer
        provider = self._build_provider_with_fees({
            EXPECTED_INDEXER: 100.0,
        })
        unknown_indexer = "0x" + "ff" * 20

        subgraph_indexers = [
            {"id": EXPECTED_INDEXER, "stakedTokens": "1000000", "lockedTokens": "0"},
            {"id": unknown_indexer, "stakedTokens": "500000", "lockedTokens": "0"},
        ]

        # Act
        with patch("redpanda.requests.post", return_value=self._mock_subgraph_response(subgraph_indexers)):
            result = provider.fetch_stake_to_fees("2024-01-01T00:00:00Z")

        # Assert
        assert result.loc[EXPECTED_INDEXER, "stake_to_fees"] == 10000.0
        assert pd.isna(result.loc[unknown_indexer, "stake_to_fees"])

    def test_output_schema(self):
        """Output must be indexed by 'indexer' with a 'stake_to_fees' column."""
        # Arrange
        provider = self._build_provider_with_fees({EXPECTED_INDEXER: 50.0})

        subgraph_indexers = [
            {"id": EXPECTED_INDEXER, "stakedTokens": "1000", "lockedTokens": "0"},
        ]

        # Act
        with patch("redpanda.requests.post", return_value=self._mock_subgraph_response(subgraph_indexers)):
            result = provider.fetch_stake_to_fees("2024-01-01T00:00:00Z")

        # Assert
        assert result.index.name == "indexer"
        assert list(result.columns) == ["stake_to_fees"]


class TestFeesAccumulatedDuringCountPass:
    """Verify that _fees_per_indexer is populated during the count pass."""

    def _build_provider(self) -> RedpandaProvider:
        with patch.dict(
            os.environ,
            {
                "REDPANDA_BOOTSTRAP_SERVERS": "localhost:9092",
                "REDPANDA_TOPIC": "gateway_queries",
            },
        ):
            return RedpandaProvider()

    def test_fees_accumulated_across_messages(self):
        """Total fees per indexer should sum across all consumed attempts."""
        # Arrange
        provider = self._build_provider()
        start_date = date(2024, 1, 1)
        num_days = 1
        base_ts_ms = int(datetime(2024, 1, 1, 12, 0, tzinfo=timezone.utc).timestamp() * 1000)

        deployment = bytes(range(32))

        # Three messages: indexer1 earns 0.001 + 0.003 = 0.004, indexer2 earns 0.002
        a1 = _build_indexer_attempt(
            indexer=INDEXER_BYTES, deployment=deployment,
            indexed_chain="mainnet", url="https://i1.example.com/",
            fee_grt=0.001, response_time_ms=50, result="success", blocks_behind=0,
        )
        a2 = _build_indexer_attempt(
            indexer=INDEXER2_BYTES, deployment=deployment,
            indexed_chain="mainnet", url="https://i2.example.com/",
            fee_grt=0.002, response_time_ms=80, result="success", blocks_behind=0,
        )
        a3 = _build_indexer_attempt(
            indexer=INDEXER_BYTES, deployment=deployment,
            indexed_chain="mainnet", url="https://i1.example.com/",
            fee_grt=0.003, response_time_ms=60, result="success", blocks_behind=0,
        )

        msg1 = _fake_kafka_message(base_ts_ms, _build_client_query("q1-JFK", [a1, a2]))
        msg2 = _fake_kafka_message(base_ts_ms + 1000, _build_client_query("q2-JFK", [a3]))

        # Resolution consumer
        resolution_consumer = _make_resolution_consumer("gateway_queries", [0])

        # Count pass consumer
        count_consumer = _make_partition_consumer([
            [msg1, msg2],
            [], [], [],
        ])

        # Act
        with patch("confluent_kafka.Consumer", side_effect=[resolution_consumer, count_consumer]):
            provider.fetch_initial_query_results(start_date, num_days)

        # Assert
        assert abs(provider._fees_per_indexer[EXPECTED_INDEXER] - 0.004) < 1e-9
        assert abs(provider._fees_per_indexer[EXPECTED_INDEXER2] - 0.002) < 1e-9
