"""Tests for progress heartbeat and stuck-detection logging in the Redpanda worker path.

Covers:
- Heartbeat from _count_partition_worker and _sample_partition_worker prints
  at PROGRESS_LOG_INTERVAL_SEC cadence.
- Heartbeat fires even when every batch is empty (regression for the prior
  behaviour where the empty-batch `continue` skipped the heartbeat check).
- Heartbeat format includes a UTC timestamp, partition id, and running counts.
- Partition resolution (_resolve_partitions) logs around list_topics and
  offsets_for_times so slow broker metadata calls are observable.
- paginate_subgraph_query logs per-page progress so slow subgraph pagination
  is observable.
"""

import logging
import re
import sys
from pathlib import Path
from unittest.mock import MagicMock, patch

import pytest

# Make the cronjob package importable.
jobs_path = Path(__file__).parent.parent / "cronjobs" / "compute_scores"
sys.path.insert(0, str(jobs_path))

import redpanda  # noqa: E402
import subgraph  # noqa: E402
from gateway_queries_pb2 import ClientQueryProtobuf  # noqa: E402
from redpanda import (  # noqa: E402
    RedpandaProvider,
    _count_partition_worker,
    _sample_partition_worker,
)


def _build_message(gateway_id: str, indexer: bytes, deployment: bytes, fee: float) -> bytes:
    query = ClientQueryProtobuf()
    query.gateway_id = gateway_id
    query.query_id = "test-query-JFK"
    attempt = query.indexer_queries.add()
    attempt.indexer = indexer
    attempt.deployment = deployment
    attempt.fee_grt = fee
    attempt.url = "https://indexer.example.com/"
    attempt.result = "success"
    return query.SerializeToString()


def _fake_kafka_message(ts_ms: int, value: bytes):
    msg = MagicMock()
    msg.error.return_value = None
    msg.timestamp.return_value = (1, ts_ms)
    msg.value.return_value = value
    msg.offset.return_value = 0
    return msg


# Matches a heartbeat line:
#   "[YYYY-MM-DDTHH:MM:SSZ <label> p<partition>] <n> msgs (<n> filtered), <n> pairs"
_HEARTBEAT_RE = re.compile(
    r"\[(\d{4}-\d{2}-\d{2}T\d{2}:\d{2}:\d{2}Z) (count|sample) p(\d+)\] "
    r"([\d,]+) msgs \(([\d,]+) filtered\), (\d+) pairs"
)
_ISO_TS_RE = re.compile(r"\d{4}-\d{2}-\d{2}T\d{2}:\d{2}:\d{2}Z")


class TestCountWorkerHeartbeat:
    """Heartbeat behaviour for _count_partition_worker."""

    def test_heartbeat_fires_with_messages(self, capsys, monkeypatch):
        """With interval=0, each loop iteration prints a heartbeat."""
        monkeypatch.setattr(redpanda, "PROGRESS_LOG_INTERVAL_SEC", 0)

        indexer = b"\x01" * 20
        deployment = b"\x02" * 32
        msg = _fake_kafka_message(1000, _build_message("mainnet-gw", indexer, deployment, 0.1))

        mock_consumer = MagicMock()
        mock_consumer.consume.side_effect = [[msg], [], [], []]

        with patch("confluent_kafka.Consumer", return_value=mock_consumer):
            _count_partition_worker(
                ("gateway_queries", 0, 0, 999999999999, {"bootstrap.servers": "x"}, None)
            )

        out = capsys.readouterr().out
        assert "[" in out and " count p0] " in out
        match = _HEARTBEAT_RE.search(out)
        assert match is not None, f"no heartbeat line matched in stdout: {out!r}"
        assert match.group(2) == "count"
        assert match.group(3) == "0"

    def test_heartbeat_fires_on_empty_batch_stretch(self, capsys, monkeypatch):
        """Regression: heartbeat must fire even when every batch is empty.

        Previously the `continue` on empty batches bypassed the heartbeat check.
        A worker sitting in a long run of 30s-timeout empty polls would appear
        completely silent until it broke out after 3 consecutive empty batches.
        """
        monkeypatch.setattr(redpanda, "PROGRESS_LOG_INTERVAL_SEC", 0)

        mock_consumer = MagicMock()
        # No messages at all — three empty batches and then the loop breaks.
        mock_consumer.consume.side_effect = [[], [], []]

        with patch("confluent_kafka.Consumer", return_value=mock_consumer):
            counts, fees, total, filtered = _count_partition_worker(
                ("gateway_queries", 0, 0, 999999999999, {"bootstrap.servers": "x"}, None)
            )

        assert total == 0
        assert filtered == 0
        assert len(counts) == 0
        assert len(fees) == 0

        out = capsys.readouterr().out
        assert " count p0] " in out, (
            f"expected a count-worker heartbeat even with only empty batches, got: {out!r}"
        )
        # Startup heartbeat plus three empty-iteration top-of-loop heartbeats
        # at interval=0: at least two lines total.
        assert out.count("count p0]") >= 2

    def test_only_startup_heartbeat_when_interval_not_reached(self, capsys, monkeypatch):
        """At the default interval, a fast run emits only the startup heartbeat."""
        monkeypatch.setattr(redpanda, "PROGRESS_LOG_INTERVAL_SEC", 120)

        indexer = b"\x01" * 20
        deployment = b"\x02" * 32
        msg = _fake_kafka_message(1000, _build_message("mainnet-gw", indexer, deployment, 0.1))

        mock_consumer = MagicMock()
        mock_consumer.consume.side_effect = [[msg], [], [], []]

        with patch("confluent_kafka.Consumer", return_value=mock_consumer):
            _count_partition_worker(
                ("gateway_queries", 0, 0, 999999999999, {"bootstrap.servers": "x"}, None)
            )

        out = capsys.readouterr().out
        matches = _HEARTBEAT_RE.findall(out)
        assert len(matches) == 1, (
            f"expected exactly the startup heartbeat at interval=120, got: {out!r}"
        )
        startup_ts, startup_label, startup_partition, startup_msgs, _, startup_pairs = matches[0]
        assert startup_label == "count"
        assert startup_partition == "0"
        # Startup heartbeat is emitted before the first consume(), so counts are zero.
        assert startup_msgs == "0"
        assert startup_pairs == "0"

    def test_heartbeat_format_has_timestamp_and_counts(self, capsys, monkeypatch):
        """Heartbeat line includes ISO-8601 UTC timestamp, msg count, filtered, pairs."""
        monkeypatch.setattr(redpanda, "PROGRESS_LOG_INTERVAL_SEC", 0)

        indexer = b"\x01" * 20
        deployment = b"\x02" * 32
        msg = _fake_kafka_message(1000, _build_message("mainnet-gw", indexer, deployment, 0.1))

        mock_consumer = MagicMock()
        mock_consumer.consume.side_effect = [[msg], [], [], []]

        with patch("confluent_kafka.Consumer", return_value=mock_consumer):
            _count_partition_worker(
                ("gateway_queries", 0, 0, 999999999999, {"bootstrap.servers": "x"}, None)
            )

        out = capsys.readouterr().out
        matches = _HEARTBEAT_RE.findall(out)
        assert matches, f"expected at least one heartbeat matching the format, got: {out!r}"
        # The LAST heartbeat (printed at top of the final empty iteration) should
        # reflect the 1 processed message.
        last_ts, last_label, last_partition, last_msgs, last_filtered, last_pairs = matches[-1]
        assert last_label == "count"
        assert last_partition == "0"
        assert last_msgs == "1"
        assert last_filtered == "0"
        assert last_pairs == "1"
        # Timestamp is YYYY-MM-DDTHH:MM:SSZ (ISO-8601 UTC).
        assert _ISO_TS_RE.fullmatch(last_ts)


class TestSampleWorkerHeartbeat:
    """Heartbeat behaviour for _sample_partition_worker (previously untested)."""

    def test_heartbeat_fires_with_messages(self, capsys, monkeypatch):
        monkeypatch.setattr(redpanda, "PROGRESS_LOG_INTERVAL_SEC", 0)

        indexer = b"\x01" * 20
        deployment = b"\x02" * 32
        msg = _fake_kafka_message(1000, _build_message("mainnet-gw", indexer, deployment, 0.1))

        mock_consumer = MagicMock()
        mock_consumer.consume.side_effect = [[msg], [], [], []]

        with patch("confluent_kafka.Consumer", return_value=mock_consumer):
            reservoirs, counts, filtered = _sample_partition_worker(
                (
                    "gateway_queries",
                    0,
                    0,
                    999999999999,
                    {"bootstrap.servers": "x"},
                    None,
                    50_000,
                    20260417,
                )
            )

        assert filtered == 0
        assert len(reservoirs) == 1
        assert len(counts) == 1

        out = capsys.readouterr().out
        matches = _HEARTBEAT_RE.findall(out)
        assert matches, f"expected sample-worker heartbeats, got: {out!r}"
        last_ts, last_label, last_partition, *_ = matches[-1]
        assert last_label == "sample"
        assert last_partition == "0"
        assert _ISO_TS_RE.fullmatch(last_ts)

    def test_heartbeat_fires_on_empty_batch_stretch(self, capsys, monkeypatch):
        """Regression: sample worker must also heartbeat on empty-only runs."""
        monkeypatch.setattr(redpanda, "PROGRESS_LOG_INTERVAL_SEC", 0)

        mock_consumer = MagicMock()
        mock_consumer.consume.side_effect = [[], [], []]

        with patch("confluent_kafka.Consumer", return_value=mock_consumer):
            reservoirs, counts, filtered = _sample_partition_worker(
                (
                    "gateway_queries",
                    0,
                    0,
                    999999999999,
                    {"bootstrap.servers": "x"},
                    None,
                    50_000,
                    20260417,
                )
            )

        assert filtered == 0
        assert len(reservoirs) == 0

        out = capsys.readouterr().out
        assert "sample p0]" in out, (
            f"expected sample-worker heartbeat on empty-only run, got: {out!r}"
        )


class TestProgressIntervalEnvOverride:
    """PROGRESS_LOG_INTERVAL_SEC is read from env at module import."""

    def test_env_var_is_honoured_on_module_load(self, monkeypatch):
        """Re-importing the module with the env set yields the overridden value."""
        import importlib

        monkeypatch.setenv("PROGRESS_LOG_INTERVAL_SEC", "7")
        reloaded = importlib.reload(redpanda)
        try:
            assert reloaded.PROGRESS_LOG_INTERVAL_SEC == 7
        finally:
            # Restore the default so subsequent tests see the normal constant.
            monkeypatch.delenv("PROGRESS_LOG_INTERVAL_SEC", raising=False)
            importlib.reload(redpanda)

    def test_default_is_120_when_env_unset(self, monkeypatch):
        import importlib

        monkeypatch.delenv("PROGRESS_LOG_INTERVAL_SEC", raising=False)
        reloaded = importlib.reload(redpanda)
        assert reloaded.PROGRESS_LOG_INTERVAL_SEC == 120


class TestResolvePartitionsLogging:
    """_resolve_partitions logs around each broker call."""

    def _make_consumer_mock(self, topic: str, partition_ids, offsets):
        partitions = {pid: MagicMock() for pid in partition_ids}
        meta = MagicMock()
        meta.topics = {topic: MagicMock(partitions=partitions)}

        resolved = []
        for pid, off in zip(partition_ids, offsets):
            tp = MagicMock()
            tp.topic = topic
            tp.partition = pid
            tp.offset = off
            resolved.append(tp)

        consumer = MagicMock()
        consumer.list_topics.return_value = meta
        consumer.offsets_for_times.return_value = resolved
        return consumer

    def test_logs_each_broker_stage(self, caplog, monkeypatch):
        """list_topics and offsets_for_times each emit an enter/exit log line."""
        caplog.set_level(logging.INFO, logger="redpanda")

        # Minimal env so RedpandaProvider() constructs cleanly.
        monkeypatch.setenv("REDPANDA_BOOTSTRAP_SERVERS", "localhost:9092")
        monkeypatch.setenv("REDPANDA_TOPIC", "gateway_queries")
        monkeypatch.delenv("REDPANDA_GATEWAY_IDS", raising=False)
        monkeypatch.delenv("IISA_PUSH_TOKEN", raising=False)

        provider = RedpandaProvider()

        consumer = self._make_consumer_mock(
            "gateway_queries", partition_ids=[0, 1, 2], offsets=[10, 20, -1]
        )
        with patch("confluent_kafka.Consumer", return_value=consumer):
            valid = provider._resolve_partitions(start_ts_ms=1_700_000_000_000)

        # One partition (offset=-1) falls out as having no data in window.
        assert len(valid) == 2

        messages = [r.getMessage() for r in caplog.records]
        assert any("list_topics(gateway_queries)" in m for m in messages), messages
        assert any("list_topics returned in" in m for m in messages), messages
        assert any("offsets_for_times across 3 partitions" in m for m in messages), messages
        assert any("offsets_for_times returned in" in m for m in messages), messages
        assert any(
            "Resolved 2 partitions with valid offsets" in m and "1 partitions had no data" in m
            for m in messages
        ), messages

    def test_raises_with_logged_context_on_missing_topic(self, caplog, monkeypatch):
        """A missing topic raises RuntimeError after the list_topics log line."""
        caplog.set_level(logging.INFO, logger="redpanda")

        monkeypatch.setenv("REDPANDA_BOOTSTRAP_SERVERS", "localhost:9092")
        monkeypatch.setenv("REDPANDA_TOPIC", "gateway_queries")

        provider = RedpandaProvider()

        meta = MagicMock()
        meta.topics = {}  # topic absent
        consumer = MagicMock()
        consumer.list_topics.return_value = meta

        with patch("confluent_kafka.Consumer", return_value=consumer):
            with pytest.raises(RuntimeError, match="Topic 'gateway_queries' not found"):
                provider._resolve_partitions(start_ts_ms=1_700_000_000_000)

        messages = [r.getMessage() for r in caplog.records]
        assert any("list_topics(gateway_queries)" in m for m in messages)


class TestSubgraphPaginationLogging:
    """paginate_subgraph_query logs per-page progress."""

    def _mock_response(self, entities):
        response = MagicMock()
        response.raise_for_status.return_value = None
        response.json.return_value = {"data": {"indexers": entities}}
        return response

    def test_logs_per_page_and_total(self, caplog):
        """Each page emits a line, and a final summary line reports totals."""
        caplog.set_level(logging.INFO, logger="subgraph")

        # Two pages, then an empty page breaks the loop. page_size=2 so the
        # full page triggers another iteration.
        responses = [
            self._mock_response([{"id": "a"}, {"id": "b"}]),
            self._mock_response([{"id": "c"}, {"id": "d"}]),
            self._mock_response([]),
        ]

        with patch("subgraph.requests.post", side_effect=responses):
            result = subgraph.paginate_subgraph_query(
                "https://subgraph.example/graphql",
                "query($first: Int!, $lastId: String!) { indexers(first: $first) { id } }",
                entity="indexers",
                page_size=2,
            )

        assert [e["id"] for e in result] == ["a", "b", "c", "d"]

        messages = [r.getMessage() for r in caplog.records]
        assert any("Paginating subgraph query" in m for m in messages), messages
        assert any("Fetched page 1" in m and "2 entities" in m for m in messages), messages
        assert any("Fetched page 2" in m and "2 entities" in m for m in messages), messages
        assert any("Fetched page 3" in m and "0 entities" in m for m in messages), messages
        assert any("Pagination complete: 4 entities across 3 page(s)" in m for m in messages), (
            messages
        )

    def test_logs_on_single_short_page(self, caplog):
        """A single partial page terminates the loop after one Fetched-page log."""
        caplog.set_level(logging.INFO, logger="subgraph")

        with patch(
            "subgraph.requests.post",
            side_effect=[self._mock_response([{"id": "only"}])],
        ):
            result = subgraph.paginate_subgraph_query(
                "https://subgraph.example/graphql",
                "query($first: Int!, $lastId: String!) { indexers(first: $first) { id } }",
                entity="indexers",
                page_size=1000,
            )

        assert [e["id"] for e in result] == ["only"]
        messages = [r.getMessage() for r in caplog.records]
        assert any("Fetched page 1" in m and "1 entities" in m for m in messages), messages
        assert any("Pagination complete: 1 entities across 1 page(s)" in m for m in messages), (
            messages
        )
