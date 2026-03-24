"""Tests for sync status loading and reverse index construction."""

import json
from datetime import datetime, timedelta, timezone

from iisa.sync_status_loader import SyncStatusData, SyncStatusLoader


def _now_iso() -> str:
    return datetime.now(timezone.utc).isoformat()


def _hours_ago_iso(hours: float) -> str:
    return (datetime.now(timezone.utc) - timedelta(hours=hours)).isoformat()


class TestSyncStatusData:
    def test_builds_deployment_index(self):
        raw = {
            "0xAAA": {
                "deployments": ["QmDeploy1", "QmDeploy2"],
                "fetched_at": _now_iso(),
            },
            "0xBBB": {
                "deployments": ["QmDeploy2", "QmDeploy3"],
                "fetched_at": _now_iso(),
            },
        }
        data = SyncStatusData(raw)

        assert data.synced_indexers_for("QmDeploy1") == {"0xaaa"}
        assert data.synced_indexers_for("QmDeploy2") == {"0xaaa", "0xbbb"}
        assert data.synced_indexers_for("QmDeploy3") == {"0xbbb"}
        assert data.total_indexers == 2
        assert data.total_deployments == 3

    def test_filters_stale_entries(self):
        raw = {
            "0xFresh": {
                "deployments": ["QmA"],
                "fetched_at": _now_iso(),
            },
            "0xStale": {
                "deployments": ["QmA", "QmB"],
                "fetched_at": _hours_ago_iso(10),
            },
        }
        data = SyncStatusData(raw, staleness_threshold_hours=6.0)

        assert data.synced_indexers_for("QmA") == {"0xfresh"}
        assert data.synced_indexers_for("QmB") == set()
        assert data.total_indexers == 1

    def test_empty_input(self):
        data = SyncStatusData({})

        assert data.synced_indexers_for("QmAnything") == set()
        assert data.total_indexers == 0
        assert data.total_deployments == 0

    def test_synced_indexers_for_unknown_deployment(self):
        raw = {
            "0xAAA": {
                "deployments": ["QmKnown"],
                "fetched_at": _now_iso(),
            },
        }
        data = SyncStatusData(raw)

        assert data.synced_indexers_for("QmUnknown") == set()

    def test_missing_fetched_at_excluded(self):
        raw = {
            "0xNoTimestamp": {
                "deployments": ["QmA"],
            },
        }
        data = SyncStatusData(raw)

        assert data.synced_indexers_for("QmA") == set()
        assert data.total_indexers == 0

    def test_empty_deployments_excluded(self):
        raw = {
            "0xEmpty": {
                "deployments": [],
                "fetched_at": _now_iso(),
            },
        }
        data = SyncStatusData(raw)

        assert data.total_indexers == 0

    def test_addresses_lowercased(self):
        raw = {
            "0xAaBbCcDd": {
                "deployments": ["QmTest"],
                "fetched_at": _now_iso(),
            },
        }
        data = SyncStatusData(raw)

        assert "0xaabbccdd" in data.synced_indexers_for("QmTest")

    def test_naive_timestamp_treated_as_utc(self):
        """Timezone-naive fetched_at is assumed UTC, not rejected."""
        raw = {
            "0xAAA": {
                "deployments": ["QmA"],
                "fetched_at": "2026-03-24T14:00:00",
            },
        }
        data = SyncStatusData(raw, staleness_threshold_hours=99999)

        assert data.synced_indexers_for("QmA") == {"0xaaa"}


class TestSyncStatusLoader:
    def test_load_success(self, tmp_path):
        path = tmp_path / "sync_status.json"
        raw = {
            "0xAAA": {
                "deployments": ["QmDeploy1"],
                "fetched_at": _now_iso(),
            },
        }
        path.write_text(json.dumps(raw))

        loader = SyncStatusLoader(str(path))
        data = loader.load()

        assert data is not None
        assert data.synced_indexers_for("QmDeploy1") == {"0xaaa"}

    def test_load_file_not_found(self, tmp_path):
        loader = SyncStatusLoader(str(tmp_path / "nonexistent.json"))
        assert loader.load() is None

    def test_load_invalid_json(self, tmp_path):
        path = tmp_path / "bad.json"
        path.write_text("not json {{{")

        loader = SyncStatusLoader(str(path))
        assert loader.load() is None

    def test_load_wrong_type(self, tmp_path):
        path = tmp_path / "array.json"
        path.write_text("[]")

        loader = SyncStatusLoader(str(path))
        assert loader.load() is None
