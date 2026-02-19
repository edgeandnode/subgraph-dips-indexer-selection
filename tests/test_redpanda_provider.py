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
from unittest.mock import MagicMock, patch

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

from redpanda import (
    MAX_RESERVOIR_PER_PAIR,
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
# Proto parser tests
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
# RedpandaProvider: _stream_and_cache via mocked Kafka
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


class TestRedpandaProviderCaching:
    """Tests for _stream_and_cache, fetch_initial_query_results, fetch_combined_query_results."""

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
        """fetch_initial_query_results should return true attempt counts."""
        # Arrange
        provider = self._build_provider()
        start_date = date(2024, 1, 1)
        num_days = 1
        ts_ms = int(datetime(2024, 1, 1, 12, 0, tzinfo=timezone.utc).timestamp() * 1000)

        deployment = bytes(range(32))
        indexer1 = INDEXER_BYTES
        indexer2 = bytes(20)  # zero bytes — 20 zeros

        msg_value = self._two_attempt_message(
            "qid-001-JFK", deployment, indexer1, indexer2, ts_ms
        )
        fake_msg = _fake_kafka_message(ts_ms, msg_value)

        # Mock the Kafka Consumer
        MockConsumer = MagicMock()
        mock_consumer = MockConsumer.return_value

        # list_topics returns a topic with one partition
        mock_topic_meta = MagicMock()
        mock_topic_meta.partitions = {0: MagicMock()}
        mock_consumer.list_topics.return_value.topics = {"gateway_queries": mock_topic_meta}

        # offsets_for_times returns a valid offset
        from confluent_kafka import TopicPartition
        mock_tp = MagicMock()
        mock_tp.topic = "gateway_queries"
        mock_tp.partition = 0
        mock_tp.offset = 100
        mock_consumer.offsets_for_times.return_value = [mock_tp]

        # poll returns one message then None (timeout sentinel)
        mock_consumer.poll.side_effect = [fake_msg, None, None, None]

        # Consumer is a local import inside _stream_and_cache; patch at source module.
        with patch("confluent_kafka.Consumer", MockConsumer):
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

        MockConsumer = MagicMock()
        mock_consumer = MockConsumer.return_value

        mock_topic_meta = MagicMock()
        mock_topic_meta.partitions = {0: MagicMock()}
        mock_consumer.list_topics.return_value.topics = {"gateway_queries": mock_topic_meta}

        mock_tp = MagicMock()
        mock_tp.topic = "gateway_queries"
        mock_tp.partition = 0
        mock_tp.offset = 0
        mock_consumer.offsets_for_times.return_value = [mock_tp]

        # Return all 5 messages then trigger timeout
        mock_consumer.poll.side_effect = messages + [None, None, None]

        # Consumer is a local import inside _stream_and_cache; patch at source module.
        with patch("confluent_kafka.Consumer", MockConsumer):
            # Fetch with rows_to_use=3 — should cap at 3 rows for this pair
            result = provider.fetch_combined_query_results(start_date, num_days, rows_to_use=3)

        # Assert
        assert len(result) == 3, f"Expected 3 rows (cap), got {len(result)}"

    def test_second_call_uses_cache(self):
        """The second fetch call must not trigger a second Kafka replay."""
        provider = self._build_provider()
        start_date = date(2024, 1, 1)
        num_days = 1

        # Seed the cache directly
        provider._cache_start_date = start_date
        provider._cache_num_days = num_days
        provider._count_cache = {(_bytes_to_cid(bytes(32)), _bytes_to_hex(INDEXER_BYTES)): 10}
        provider._row_cache_df = pd.DataFrame(
            [{"query_id": "x", "deployment_hash": _bytes_to_cid(bytes(32)),
              "fee": 0.001, "timestamp": pd.Timestamp("2024-01-01", tz="UTC"),
              "blocks_behind": 0, "response_time_ms": 100,
              "indexer": _bytes_to_hex(INDEXER_BYTES), "status": "200 OK",
              "day_partition": date(2024, 1, 1), "subgraph_network": "mainnet",
              "url": "https://indexer.example.com/"}]
        )

        with patch.object(provider, "_stream_and_cache") as mock_stream:
            result = provider.fetch_initial_query_results(start_date, num_days)

        # _stream_and_cache must NOT have been called when cache is valid
        mock_stream.assert_not_called()
        assert len(result) == 1


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

    def test_output_schema_matches_bigquery(self):
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


class TestFeesAccumulatedDuringConsume:
    """Verify that _fees_per_indexer is populated during the Kafka replay."""

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

        MockConsumer = MagicMock()
        mock_consumer = MockConsumer.return_value

        mock_topic_meta = MagicMock()
        mock_topic_meta.partitions = {0: MagicMock()}
        mock_consumer.list_topics.return_value.topics = {"gateway_queries": mock_topic_meta}

        mock_tp = MagicMock()
        mock_tp.topic = "gateway_queries"
        mock_tp.partition = 0
        mock_tp.offset = 0
        mock_consumer.offsets_for_times.return_value = [mock_tp]

        mock_consumer.poll.side_effect = [msg1, msg2, None, None, None]

        # Act
        with patch("confluent_kafka.Consumer", MockConsumer):
            provider.fetch_initial_query_results(start_date, num_days)

        # Assert
        assert abs(provider._fees_per_indexer[EXPECTED_INDEXER] - 0.004) < 1e-9
        assert abs(provider._fees_per_indexer[EXPECTED_INDEXER2] - 0.002) < 1e-9
