"""Tests for the sync status fetcher service."""

import json
import sys
from pathlib import Path
from unittest.mock import AsyncMock, MagicMock, patch

import pytest

# Make the cronjob package importable.
jobs_path = Path(__file__).parent.parent / "cronjobs" / "compute_scores"
sys.path.insert(0, str(jobs_path))

from sync_status_fetcher import (  # noqa: E402
    _fetch_all_statuses,
    _fetch_single_status,
    _write_sync_status,
    run_fetch_cycle,
)


class TestFetchSingleStatus:
    """Tests for fetching and filtering a single indexer's /status."""

    @pytest.fixture
    def mock_response(self):
        """Build a mock aiohttp response with configurable JSON."""

        def _make(json_data, status=200):
            resp = AsyncMock()
            resp.status = status
            resp.json = AsyncMock(return_value=json_data)
            resp.raise_for_status = MagicMock()
            resp.__aenter__ = AsyncMock(return_value=resp)
            resp.__aexit__ = AsyncMock(return_value=False)
            return resp

        return _make

    @pytest.mark.asyncio
    async def test_filters_to_synced_healthy_only(self, mock_response):
        """Only deployments with synced=true AND health=healthy pass."""
        json_data = {
            "data": {
                "indexingStatuses": [
                    {"subgraph": "QmSynced", "synced": True, "health": "healthy"},
                    {"subgraph": "QmUnhealthy", "synced": True, "health": "unhealthy"},
                    {"subgraph": "QmNotSynced", "synced": False, "health": "healthy"},
                    {"subgraph": "QmFailed", "synced": False, "health": "failed"},
                ]
            }
        }
        resp = mock_response(json_data)
        session = MagicMock()
        session.post = MagicMock(return_value=resp)
        semaphore = MagicMock()
        semaphore.__aenter__ = AsyncMock()
        semaphore.__aexit__ = AsyncMock()

        result = await _fetch_single_status(
            session, "0xAAA", "https://indexer.example.com", semaphore
        )

        assert result is not None
        assert result["indexer"] == "0xAAA"
        assert result["deployments"] == ["QmSynced"]

    @pytest.mark.asyncio
    async def test_returns_none_on_persistent_failure(self, mock_response):
        """Returns None after all retries exhausted."""
        import asyncio

        # The error must come from inside the async context manager.
        # __aexit__ must return False or the exception gets suppressed.
        failing_cm = AsyncMock()
        failing_cm.__aenter__ = AsyncMock(side_effect=asyncio.TimeoutError("timeout"))
        failing_cm.__aexit__ = AsyncMock(return_value=False)
        session = MagicMock()
        session.post = MagicMock(return_value=failing_cm)
        semaphore = MagicMock()
        semaphore.__aenter__ = AsyncMock()
        semaphore.__aexit__ = AsyncMock(return_value=False)

        with patch("sync_status_fetcher.MAX_RETRIES", 1):
            result = await _fetch_single_status(
                session, "0xBBB", "https://dead.example.com", semaphore
            )

        assert result is None

    @pytest.mark.asyncio
    async def test_empty_statuses_returns_empty_deployments(self, mock_response):
        """Indexer with no deployments returns empty list."""
        json_data = {"data": {"indexingStatuses": []}}
        resp = mock_response(json_data)
        session = MagicMock()
        session.post = MagicMock(return_value=resp)
        semaphore = MagicMock()
        semaphore.__aenter__ = AsyncMock()
        semaphore.__aexit__ = AsyncMock()

        result = await _fetch_single_status(
            session, "0xCCC", "https://empty.example.com", semaphore
        )

        assert result is not None
        assert result["deployments"] == []


class TestFetchAllStatuses:
    """Tests for concurrent fetching across multiple indexers."""

    @pytest.mark.asyncio
    async def test_excludes_indexers_with_no_synced_deployments(self):
        """Indexers with empty deployment lists are excluded from output."""

        async def mock_fetch(session, indexer, url, sem):
            if indexer == "0xGood":
                return {"indexer": "0xGood", "deployments": ["QmA"]}
            return {"indexer": "0xEmpty", "deployments": []}

        with patch(
            "sync_status_fetcher._fetch_single_status",
            side_effect=mock_fetch,
        ):
            result = await _fetch_all_statuses(
                {
                    "0xGood": "https://good.com",
                    "0xEmpty": "https://empty.com",
                }
            )

        assert "0xGood" in result
        assert "0xEmpty" not in result
        assert "fetched_at" in result["0xGood"]

    @pytest.mark.asyncio
    async def test_excludes_failed_fetches(self):
        """Indexers that returned None are excluded."""

        async def mock_fetch(session, indexer, url, sem):
            if indexer == "0xOK":
                return {"indexer": "0xOK", "deployments": ["QmA"]}
            return None

        with patch(
            "sync_status_fetcher._fetch_single_status",
            side_effect=mock_fetch,
        ):
            result = await _fetch_all_statuses(
                {
                    "0xOK": "https://ok.com",
                    "0xDead": "https://dead.com",
                }
            )

        assert "0xOK" in result
        assert "0xDead" not in result


class TestWriteSyncStatus:
    """Tests for atomic file writing."""

    def test_writes_valid_json(self, tmp_path):
        data = {"0xAAA": {"deployments": ["QmA"], "fetched_at": "now"}}

        with patch(
            "sync_status_fetcher.SYNC_STATUS_FILE_PATH",
            str(tmp_path / "sync_status.json"),
        ):
            _write_sync_status(data)

        written = json.loads((tmp_path / "sync_status.json").read_text())
        assert written == data

    def test_no_tmp_file_left_behind(self, tmp_path):
        data = {"0xAAA": {"deployments": ["QmA"], "fetched_at": "now"}}
        out = str(tmp_path / "sync_status.json")

        with patch("sync_status_fetcher.SYNC_STATUS_FILE_PATH", out):
            _write_sync_status(data)

        assert not Path(out + ".tmp").exists()
        assert Path(out).exists()


class TestRunFetchCycle:
    """Tests for the full fetch-filter-write pipeline."""

    def test_writes_file_on_success(self, tmp_path):
        out = str(tmp_path / "sync_status.json")
        indexer_urls = {"0xAAA": "https://indexer.example.com"}
        fetch_result = {
            "0xAAA": {
                "deployments": ["QmDeploy1"],
                "fetched_at": "2026-03-24T00:00:00+00:00",
            }
        }

        with (
            patch(
                "sync_status_fetcher.discover_indexers_from_network_subgraph",
                return_value=indexer_urls,
            ),
            patch(
                "sync_status_fetcher.asyncio.run",
                return_value=fetch_result,
            ),
            patch("sync_status_fetcher.SYNC_STATUS_FILE_PATH", out),
            patch("sync_status_fetcher._notify_iisa"),
        ):
            result = run_fetch_cycle()

        assert result is True
        written = json.loads(Path(out).read_text())
        assert "0xAAA" in written

    def test_skips_when_no_indexers(self):
        with patch(
            "sync_status_fetcher.discover_indexers_from_network_subgraph",
            return_value={},
        ):
            result = run_fetch_cycle()

        assert result is False
