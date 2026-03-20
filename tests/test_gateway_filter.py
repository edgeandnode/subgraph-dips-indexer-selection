"""Tests for the REDPANDA_GATEWAY_IDS filter on RedpandaProvider."""

import os
import sys
from pathlib import Path
from unittest.mock import MagicMock, patch

# Make the cronjob package importable.
jobs_path = Path(__file__).parent.parent / "cronjobs" / "compute_scores"
sys.path.insert(0, str(jobs_path))

from redpanda import RedpandaProvider  # noqa: E402


class TestGatewayIdFilter:
    """Test the gateway_id filter initialization and message filtering."""

    def test_filter_active_when_env_set(self):
        """When REDPANDA_GATEWAY_IDS is set, only those gateways are allowed."""
        with patch.dict(os.environ, {"REDPANDA_GATEWAY_IDS": "gw-1,gw-2"}):
            provider = RedpandaProvider()
        assert provider._gateway_id_filter == {"gw-1", "gw-2"}

    def test_filter_none_when_env_unset(self):
        """When REDPANDA_GATEWAY_IDS is not set, no filtering occurs."""
        with patch.dict(os.environ, {}, clear=False):
            env = os.environ.copy()
            env.pop("REDPANDA_GATEWAY_IDS", None)
            with patch.dict(os.environ, env, clear=True):
                provider = RedpandaProvider()
        assert provider._gateway_id_filter is None

    def test_filter_none_when_env_empty(self):
        """Empty string is treated the same as unset."""
        with patch.dict(os.environ, {"REDPANDA_GATEWAY_IDS": ""}):
            provider = RedpandaProvider()
        assert provider._gateway_id_filter is None

    def test_filter_strips_whitespace(self):
        """Whitespace around gateway IDs is stripped."""
        with patch.dict(os.environ, {"REDPANDA_GATEWAY_IDS": " gw-1 , gw-2 "}):
            provider = RedpandaProvider()
        assert provider._gateway_id_filter == {"gw-1", "gw-2"}

    def test_filter_ignores_empty_segments(self):
        """Trailing commas or double commas don't create empty entries."""
        with patch.dict(os.environ, {"REDPANDA_GATEWAY_IDS": "gw-1,,gw-2,"}):
            provider = RedpandaProvider()
        assert provider._gateway_id_filter == {"gw-1", "gw-2"}

    def test_count_loop_skips_non_matching_gateway(self):
        """Messages from non-matching gateways are skipped in the count loop."""
        with patch.dict(os.environ, {"REDPANDA_GATEWAY_IDS": "mainnet-gw"}):
            provider = RedpandaProvider()

        # Mock a protobuf message from the wrong gateway
        mock_query = MagicMock()
        mock_query.gateway_id = "testnet-gw"
        mock_query.indexer_queries = [MagicMock()]

        # Mock consumer that returns one message then stops
        mock_msg = MagicMock()
        mock_msg.error.return_value = None
        mock_msg.timestamp.return_value = (0, 1000)
        mock_msg.value.return_value = b"fake"

        mock_consumer = MagicMock()
        mock_consumer.consume.side_effect = [[mock_msg], [], [], []]

        with patch("redpanda.ClientQueryProtobuf") as MockProto:
            instance = MockProto.return_value
            instance.ParseFromString.return_value = None
            instance.gateway_id = "testnet-gw"
            instance.indexer_queries = [
                MagicMock(indexer=b"\x01" * 20, deployment=b"\x02" * 32, fee_grt=0.1)
            ]

            counts, fees, total, filtered = provider._count_partition_loop(
                mock_consumer, end_ts_ms=999999999999
            )

        # Message was consumed (total_messages incremented) but indexer data was skipped
        assert total == 1
        assert filtered == 1
        assert len(counts) == 0
        assert len(fees) == 0

    def test_count_loop_accepts_matching_gateway(self):
        """Messages from matching gateways are processed in the count loop."""
        with patch.dict(os.environ, {"REDPANDA_GATEWAY_IDS": "mainnet-gw"}):
            provider = RedpandaProvider()

        mock_msg = MagicMock()
        mock_msg.error.return_value = None
        mock_msg.timestamp.return_value = (0, 1000)
        mock_msg.value.return_value = b"fake"

        mock_consumer = MagicMock()
        mock_consumer.consume.side_effect = [[mock_msg], [], [], []]

        with patch("redpanda.ClientQueryProtobuf") as MockProto:
            instance = MockProto.return_value
            instance.ParseFromString.return_value = None
            instance.gateway_id = "mainnet-gw"
            instance.indexer_queries = [
                MagicMock(indexer=b"\x01" * 20, deployment=b"\x02" * 32, fee_grt=0.1)
            ]

            counts, fees, total, filtered = provider._count_partition_loop(
                mock_consumer, end_ts_ms=999999999999
            )

        assert total == 1
        assert filtered == 0
        assert len(counts) == 1
        assert len(fees) == 1

    def test_count_loop_no_filter_passes_all(self):
        """When no filter is set, all messages are processed."""
        with patch.dict(os.environ, {"REDPANDA_GATEWAY_IDS": ""}):
            provider = RedpandaProvider()

        assert provider._gateway_id_filter is None

        mock_msg = MagicMock()
        mock_msg.error.return_value = None
        mock_msg.timestamp.return_value = (0, 1000)
        mock_msg.value.return_value = b"fake"

        mock_consumer = MagicMock()
        mock_consumer.consume.side_effect = [[mock_msg], [], [], []]

        with patch("redpanda.ClientQueryProtobuf") as MockProto:
            instance = MockProto.return_value
            instance.ParseFromString.return_value = None
            instance.gateway_id = "any-gateway"
            instance.indexer_queries = [
                MagicMock(indexer=b"\x01" * 20, deployment=b"\x02" * 32, fee_grt=0.1)
            ]

            counts, fees, total, filtered = provider._count_partition_loop(
                mock_consumer, end_ts_ms=999999999999
            )

        assert total == 1
        assert filtered == 0
        assert len(counts) == 1
