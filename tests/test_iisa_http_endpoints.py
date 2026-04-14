"""Tests for the IISA HTTP API endpoints."""

import json
import os
from datetime import datetime, timezone
from unittest.mock import MagicMock, patch

import pandas as pd
import pytest
from fastapi.testclient import TestClient


class TestSettings:
    """Tests for Settings class and get_settings()."""

    def test_settings_loads_from_env(self):
        """Verify IISA_ prefix environment variables load correctly."""
        # Arrange
        env_vars = {
            "IISA_HOST": "127.0.0.1",
            "IISA_PORT": "9000",
            "IISA_LOG_LEVEL": "DEBUG",
        }

        # Act
        with patch.dict(os.environ, env_vars, clear=False):
            from iisa.iisa_http_endpoints import Settings

            settings = Settings()

        # Assert
        assert settings.host == "127.0.0.1"
        assert settings.port == 9000
        assert settings.log_level == "DEBUG"

    def test_settings_default_values(self):
        """Verify defaults: host="0.0.0.0", port=8080, log_level="INFO"."""
        # Arrange & Act
        from iisa.iisa_http_endpoints import Settings

        settings = Settings()

        # Assert
        assert settings.host == "0.0.0.0"
        assert settings.port == 8080
        assert settings.log_level == "INFO"

    def test_get_settings_cached(self):
        """Verify @lru_cache returns same instance."""
        # Arrange
        env_vars = {}

        with patch.dict(os.environ, env_vars, clear=False):
            # Need to reimport to get fresh cache
            from iisa import iisa_http_endpoints

            # Clear the cache first
            iisa_http_endpoints.get_settings.cache_clear()

            # Act
            settings1 = iisa_http_endpoints.get_settings()
            settings2 = iisa_http_endpoints.get_settings()

            # Assert
            assert settings1 is settings2


class TestPydanticModels:
    """Tests for request/response model validation."""

    def test_selection_request_requires_num_candidates(self):
        """num_candidates is required along with deployment_id."""
        # Arrange & Act
        from iisa.iisa_http_endpoints import SelectionRequest

        request = SelectionRequest(deployment_id="Qm123", num_candidates=3)

        # Assert
        assert request.deployment_id == "Qm123"
        assert request.num_candidates == 3
        assert request.existing_indexers is None
        assert request.pending_agreements is None
        assert request.blocklist is None
        assert request.declined_indexers is None

    def test_selection_request_with_all_fields(self):
        """Verify all optional fields serialize correctly."""
        # Arrange & Act
        from iisa.iisa_http_endpoints import SelectionRequest

        request = SelectionRequest(
            deployment_id="Qm123",
            existing_indexers=["0x111"],
            num_candidates=2,
            blocklist=["0xBAD"],
            pending_agreements={"Qm123": ["0xPEND"]},
            declined_indexers={"Qm123": ["0xDEC"]},
        )

        # Assert
        assert request.existing_indexers == ["0x111"]
        assert request.num_candidates == 2
        assert request.blocklist == ["0xBAD"]
        assert request.pending_agreements == {"Qm123": ["0xPEND"]}
        assert request.declined_indexers == {"Qm123": ["0xDEC"]}

    def test_selection_request_missing_num_candidates_raises(self):
        """Verify ValidationError when num_candidates missing."""
        # Arrange & Act & Assert
        from pydantic import ValidationError as PydanticValidationError

        from iisa.iisa_http_endpoints import SelectionRequest

        with pytest.raises(PydanticValidationError) as exc_info:
            SelectionRequest(deployment_id="Qm123")

        assert "num_candidates" in str(exc_info.value)

    def test_selection_response(self):
        """deployment_id and indexers fields."""
        # Arrange & Act
        from iisa.iisa_http_endpoints import SelectedIndexer, SelectionResponse

        response = SelectionResponse(
            deployment_id="Qm123",
            indexers=[
                SelectedIndexer(id="0xABC", min_grt_per_30_days=450.0),
                SelectedIndexer(id="0xXYZ"),
                SelectedIndexer(id="0x123"),
            ],
        )

        # Assert
        assert response.deployment_id == "Qm123"
        assert len(response.indexers) == 3
        assert response.indexers[0].id == "0xABC"
        assert response.indexers[0].min_grt_per_30_days == 450.0

    def test_selection_response_empty_indexers(self):
        """Verify empty indexers list is valid."""
        # Arrange & Act
        from iisa.iisa_http_endpoints import SelectionResponse

        response = SelectionResponse(deployment_id="Qm123", indexers=[])

        # Assert
        assert response.deployment_id == "Qm123"
        assert response.indexers == []

    def test_health_response(self):
        """status and data_loaded fields."""
        # Arrange & Act
        from iisa.iisa_http_endpoints import HealthResponse

        response = HealthResponse(status="healthy", data_loaded=True)

        # Assert
        assert response.status == "healthy"
        assert response.data_loaded is True


class TestIISAState:
    """Tests for IISAState lifecycle management."""

    def test_init_default_state(self):
        """Verify initial state has all fields set to None/False."""
        # Arrange & Act
        from iisa.iisa_http_endpoints import IISAState

        state = IISAState()

        # Assert
        assert state.settings is None
        assert state.data_manager is None
        assert state._history is None
        assert state._initialized is False

    @patch("iisa.iisa_http_endpoints.DataManager")
    @patch("iisa.iisa_http_endpoints.FileScoreLoader")
    def test_initialize_success(self, mock_loader_class, mock_dm_class):
        """Mock FileScoreLoader/DataManager, verify _initialized=True."""
        # Arrange
        from iisa.iisa_http_endpoints import IISAState, Settings

        mock_loader_instance = MagicMock()
        mock_loader_class.return_value = mock_loader_instance

        mock_dm_instance = MagicMock()
        mock_dm_class.return_value = mock_dm_instance

        state = IISAState()
        settings = Settings()

        # Act
        result = state.initialize(settings)

        # Assert
        assert result is True
        assert state._initialized is True
        assert state.settings is settings
        assert state.data_manager is mock_dm_instance
        mock_loader_class.assert_called_once()
        mock_dm_class.assert_called_once_with(mock_loader_instance)

    @patch("iisa.iisa_http_endpoints.FileScoreLoader")
    def test_initialize_failure(self, mock_loader_class):
        """Mock FileScoreLoader to raise, verify returns False and logs warning."""
        # Arrange
        from iisa.iisa_http_endpoints import IISAState, Settings

        mock_loader_class.side_effect = Exception("Connection failed")

        state = IISAState()
        settings = Settings()

        # Act
        result = state.initialize(settings)

        # Assert
        assert result is False
        assert state._initialized is False

    @patch("iisa.iisa_http_endpoints.DataManager")
    @patch("iisa.iisa_http_endpoints.FileScoreLoader")
    def test_refresh_data_success(self, mock_loader_class, mock_dm_class):
        """Mock load_scores()=True, verify _history populated."""
        # Arrange
        from iisa.iisa_http_endpoints import IISAState, Settings

        mock_history_df = pd.DataFrame({"indexer": ["0xABC", "0xXYZ"]})

        mock_dm_instance = MagicMock()
        mock_dm_instance.load_scores.return_value = True
        mock_dm_instance.get_data.return_value = mock_history_df
        mock_dm_class.return_value = mock_dm_instance

        state = IISAState()
        settings = Settings()
        state.initialize(settings)

        # Act
        result = state.refresh_data()

        # Assert
        assert result is True
        assert state._history is not None
        assert len(state._history) == 2
        mock_dm_instance.load_scores.assert_called_once()

    @patch("iisa.iisa_http_endpoints.DataManager")
    @patch("iisa.iisa_http_endpoints.FileScoreLoader")
    def test_refresh_data_failure(self, mock_loader_class, mock_dm_class):
        """Mock load_scores()=False, verify returns False."""
        # Arrange
        from iisa.iisa_http_endpoints import IISAState, Settings

        mock_dm_instance = MagicMock()
        mock_dm_instance.load_scores.return_value = False
        mock_dm_class.return_value = mock_dm_instance

        state = IISAState()
        settings = Settings()
        state.initialize(settings)

        # Act
        result = state.refresh_data()

        # Assert
        assert result is False

    @patch("iisa.iisa_http_endpoints.DataManager")
    @patch("iisa.iisa_http_endpoints.FileScoreLoader")
    def test_refresh_data_exception(self, mock_loader_class, mock_dm_class):
        """Mock load_scores() to raise exception, verify returns False."""
        # Arrange
        from iisa.iisa_http_endpoints import IISAState, Settings

        mock_dm_instance = MagicMock()
        mock_dm_instance.load_scores.side_effect = Exception("Connection failed")
        mock_dm_class.return_value = mock_dm_instance

        state = IISAState()
        settings = Settings()
        state.initialize(settings)

        # Act
        result = state.refresh_data()

        # Assert
        assert result is False

    def test_refresh_data_not_initialized(self):
        """Call refresh before initialize, verify returns False."""
        # Arrange
        from iisa.iisa_http_endpoints import IISAState

        state = IISAState()

        # Act
        result = state.refresh_data()

        # Assert
        assert result is False

    def test_is_ready_with_data(self):
        """Set _history to non-empty DataFrame, verify is_ready=True."""
        # Arrange
        from iisa.iisa_http_endpoints import IISAState

        state = IISAState()
        state._history = pd.DataFrame({"indexer": ["0xABC"]})

        # Act & Assert
        assert state.is_ready is True

    def test_is_ready_without_data(self):
        """Verify is_ready=False when _history is None or empty."""
        # Arrange
        from iisa.iisa_http_endpoints import IISAState

        state = IISAState()

        # Act & Assert - None case
        assert state.is_ready is False

        # Arrange - empty DataFrame case
        state._history = pd.DataFrame()

        # Act & Assert
        assert state.is_ready is False


@pytest.fixture(autouse=True)
def reset_state(monkeypatch):
    """Reset global state and clean push-token env vars before each test.

    The module-level _state singleton and the lru_cache on get_settings
    otherwise leak between tests. Env vars set via monkeypatch.setenv in
    one test (e.g. IISA_PUSH_TOKEN) must not be visible to the next.
    """
    from iisa import iisa_http_endpoints

    monkeypatch.delenv("IISA_PUSH_TOKEN", raising=False)
    monkeypatch.delenv("IISA_REQUIRE_PUSH_TOKEN", raising=False)
    iisa_http_endpoints.get_settings.cache_clear()
    iisa_http_endpoints._state = iisa_http_endpoints.IISAState()

    yield

    iisa_http_endpoints._state = iisa_http_endpoints.IISAState()
    iisa_http_endpoints.get_settings.cache_clear()


@pytest.fixture
def mock_history_df():
    """Create a mock DataFrame simulating loaded history data."""
    return pd.DataFrame(
        {
            "indexer": ["0xabc", "0xxyz", "0x123"],  # lowercase for case-insensitive matching
            "url": ["https://a.com/", "https://b.com/", "https://c.com/"],
            "norm_lat_lin_reg_coefficient": [0.8, 0.9, 0.6],
            "norm_uptime_score": [0.9, 0.7, 0.95],
            "norm_success_rate": [0.85, 0.6, 0.9],
            "norm_stake_to_fees": [0.5, 0.8, 0.65],
            "norm_base_price_per_epoch": [0.7, 0.8, 0.9],
            "norm_price_per_entity": [0.6, 0.7, 0.8],
            "dips_info_available": [True, True, False],
            "dips_min_grt_per_30_days": [
                '{"arbitrum-one": "450"}',
                '{"arbitrum-one": "500"}',
                "{}",
            ],
            "dips_min_grt_per_billion_entities_per_30_days": ["200", "300", None],
            "dips_supported_networks": ['["arbitrum-one"]', '["arbitrum-one"]', "[]"],
        }
    )


class TestHealthEndpoint:
    """Tests for GET /health endpoint."""

    def test_health_with_data_loaded(self, mock_history_df):
        """is_ready=True returns data_loaded=True."""
        # Arrange
        from iisa import iisa_http_endpoints
        from iisa.iisa_http_endpoints import app

        iisa_http_endpoints._state._history = mock_history_df
        client = TestClient(app, raise_server_exceptions=False)

        # Act
        response = client.get("/health")

        # Assert
        assert response.status_code == 200
        data = response.json()
        assert data["status"] == "healthy"
        assert data["data_loaded"] is True

    def test_health_without_data(self):
        """is_ready=False returns data_loaded=False."""
        # Arrange
        from iisa.iisa_http_endpoints import app

        client = TestClient(app, raise_server_exceptions=False)

        # Act
        response = client.get("/health")

        # Assert
        assert response.status_code == 200
        data = response.json()
        assert data["status"] == "healthy"
        assert data["data_loaded"] is False


class TestPushScoresEndpoint:
    """Tests for POST /scores endpoint (cronjob → iisa push)."""

    @staticmethod
    def _sample_payload():
        return [
            {
                "indexer": "0xABC",
                "computed_at": "2026-04-14T09:00:00+00:00",
                "lat_normalized_score": 0.8,
                "uptime_score": 0.95,
                "success_rate": 0.99,
            }
        ]

    @patch("iisa.iisa_http_endpoints.DataManager")
    @patch("iisa.iisa_http_endpoints.FileScoreLoader")
    def test_push_scores_success(self, mock_loader_class, mock_dm_class, tmp_path, monkeypatch):
        """Valid body + no auth required (unset token) → 200 with row count."""
        from iisa import iisa_http_endpoints
        from iisa.iisa_http_endpoints import Settings, app

        monkeypatch.setattr(iisa_http_endpoints, "SCORES_FILE_PATH", str(tmp_path / "scores.json"))

        mock_dm_instance = MagicMock()
        mock_dm_instance.load_scores_from_df.return_value = True
        mock_dm_instance.get_data.return_value = pd.DataFrame({"indexer": ["0xABC"]})
        mock_dm_class.return_value = mock_dm_instance

        settings = Settings()  # push_token unset in test env
        iisa_http_endpoints._state.initialize(settings)

        client = TestClient(app, raise_server_exceptions=False)
        response = client.post("/scores", json=self._sample_payload())

        assert response.status_code == 200
        body = response.json()
        assert body["status"] == "success"
        assert body["rows"] == 1
        # Disk was written atomically
        assert (tmp_path / "scores.json").exists()

    def test_push_scores_empty_body_rejected(self):
        """Empty payload → 422."""
        from iisa import iisa_http_endpoints
        from iisa.iisa_http_endpoints import Settings, app

        iisa_http_endpoints._state.initialize(Settings())
        client = TestClient(app, raise_server_exceptions=False)

        response = client.post("/scores", json=[])
        assert response.status_code == 422

    def test_push_scores_rejects_missing_token_when_required(self, monkeypatch):
        """IISA_PUSH_TOKEN set + no header → 401."""
        monkeypatch.setenv("IISA_PUSH_TOKEN", "secret")

        from iisa import iisa_http_endpoints
        from iisa.iisa_http_endpoints import Settings, app

        iisa_http_endpoints.get_settings.cache_clear()
        iisa_http_endpoints._state.initialize(Settings())
        client = TestClient(app, raise_server_exceptions=False)

        response = client.post("/scores", json=self._sample_payload())
        assert response.status_code == 401
        assert response.json()["detail"] == "Missing or malformed Authorization header"

    def test_push_scores_rejects_wrong_token(self, monkeypatch):
        """IISA_PUSH_TOKEN set + wrong bearer → 401."""
        monkeypatch.setenv("IISA_PUSH_TOKEN", "secret")

        from iisa import iisa_http_endpoints
        from iisa.iisa_http_endpoints import Settings, app

        iisa_http_endpoints.get_settings.cache_clear()
        iisa_http_endpoints._state.initialize(Settings())
        client = TestClient(app, raise_server_exceptions=False)

        response = client.post(
            "/scores",
            json=self._sample_payload(),
            headers={"Authorization": "Bearer wrong"},
        )
        assert response.status_code == 401
        assert "Invalid" in response.json()["detail"]

    @patch("iisa.iisa_http_endpoints.DataManager")
    @patch("iisa.iisa_http_endpoints.FileScoreLoader")
    def test_push_scores_transform_failure_does_not_touch_disk(
        self, mock_loader_class, mock_dm_class, tmp_path, monkeypatch
    ):
        """
        Dry-run invariant: if transform_scores_df raises, the cache file
        at SCORES_FILE_PATH must NOT be written. A restart should still
        load the previous valid payload, not a poisoned one.
        """
        from iisa import iisa_http_endpoints
        from iisa.iisa_http_endpoints import Settings, app

        scores_path = tmp_path / "scores.json"
        monkeypatch.setattr(iisa_http_endpoints, "SCORES_FILE_PATH", str(scores_path))

        # Transform raises — simulates a schema-invalid payload.
        mock_dm_instance = MagicMock()
        mock_dm_instance.transform_scores_df.side_effect = KeyError("missing column")
        mock_dm_class.return_value = mock_dm_instance

        iisa_http_endpoints._state.initialize(Settings())
        client = TestClient(app, raise_server_exceptions=False)

        response = client.post("/scores", json=self._sample_payload())

        # Handler 500s on the transform failure.
        assert response.status_code == 500
        # Critical invariant: disk was never touched. commit_scores was also
        # never called — memory and disk both untouched by the failed push.
        assert not scores_path.exists()
        mock_dm_instance.commit_scores.assert_not_called()

    @patch("iisa.iisa_http_endpoints.DataManager")
    @patch("iisa.iisa_http_endpoints.FileScoreLoader")
    def test_push_scores_accepts_valid_token(
        self, mock_loader_class, mock_dm_class, tmp_path, monkeypatch
    ):
        """IISA_PUSH_TOKEN set + matching bearer → 200."""
        from iisa import iisa_http_endpoints
        from iisa.iisa_http_endpoints import Settings, app

        monkeypatch.setenv("IISA_PUSH_TOKEN", "secret")
        monkeypatch.setattr(iisa_http_endpoints, "SCORES_FILE_PATH", str(tmp_path / "scores.json"))
        iisa_http_endpoints.get_settings.cache_clear()

        mock_dm_instance = MagicMock()
        mock_dm_instance.load_scores_from_df.return_value = True
        mock_dm_instance.get_data.return_value = pd.DataFrame({"indexer": ["0xABC"]})
        mock_dm_class.return_value = mock_dm_instance

        iisa_http_endpoints._state.initialize(Settings())
        client = TestClient(app, raise_server_exceptions=False)

        response = client.post(
            "/scores",
            json=self._sample_payload(),
            headers={"Authorization": "Bearer secret"},
        )
        assert response.status_code == 200


class TestScoresStatusEndpoint:
    """Tests for GET /scores/status endpoint (cronjob idempotency check)."""

    def test_returns_computed_at_when_loaded(self, monkeypatch):
        from iisa import iisa_http_endpoints
        from iisa.iisa_http_endpoints import Settings, app

        iisa_http_endpoints.get_settings.cache_clear()
        iisa_http_endpoints._state.initialize(Settings())

        # Seed state with a fake DataManager that has a computed_at
        mock_dm = MagicMock()
        mock_dm._scores_computed_at = datetime(2026, 4, 14, 9, 0, tzinfo=timezone.utc)
        iisa_http_endpoints._state.data_manager = mock_dm
        iisa_http_endpoints._state._history = pd.DataFrame({"indexer": ["0xABC"]})

        client = TestClient(app, raise_server_exceptions=False)
        response = client.get("/scores/status")

        assert response.status_code == 200
        body = response.json()
        assert body["computed_at"] is not None
        assert "2026-04-14" in body["computed_at"]
        assert body["rows"] == 1

    def test_returns_null_when_no_scores_loaded(self):
        from iisa import iisa_http_endpoints
        from iisa.iisa_http_endpoints import Settings, app

        iisa_http_endpoints._state.initialize(Settings())
        client = TestClient(app, raise_server_exceptions=False)

        response = client.get("/scores/status")
        assert response.status_code == 200
        body = response.json()
        assert body["computed_at"] is None
        assert body["rows"] == 0


class TestGetScoreEndpoint:
    """Tests for POST /get-score endpoint."""

    @patch("iisa.iisa_http_endpoints.DataManager")
    @patch("iisa.iisa_http_endpoints.FileScoreLoader")
    def test_get_score_found(self, mock_loader_class, mock_dm_class, mock_history_df):
        """Verify returns score and components for existing indexer."""
        # Arrange
        from iisa import iisa_http_endpoints
        from iisa.iisa_http_endpoints import Settings, app

        mock_dm_instance = MagicMock()
        mock_dm_instance.load_scores.return_value = True
        mock_dm_instance.get_data.return_value = mock_history_df
        mock_dm_class.return_value = mock_dm_instance

        settings = Settings()
        iisa_http_endpoints._state.initialize(settings)
        iisa_http_endpoints._state.refresh_data()

        client = TestClient(app, raise_server_exceptions=False)

        # Act
        response = client.post("/get-score", json={"indexer_id": "0xABC"})

        # Assert
        assert response.status_code == 200
        data = response.json()
        assert data["indexer_id"] == "0xABC"
        assert data["found"] is True
        assert data["weighted_score"] is not None
        assert isinstance(data["components"], dict)

    @patch("iisa.iisa_http_endpoints.DataManager")
    @patch("iisa.iisa_http_endpoints.FileScoreLoader")
    def test_get_score_not_found(self, mock_loader_class, mock_dm_class, mock_history_df):
        """Verify returns found=false for non-existent indexer."""
        # Arrange
        from iisa import iisa_http_endpoints
        from iisa.iisa_http_endpoints import Settings, app

        mock_dm_instance = MagicMock()
        mock_dm_instance.load_scores.return_value = True
        mock_dm_instance.get_data.return_value = mock_history_df
        mock_dm_class.return_value = mock_dm_instance

        settings = Settings()
        iisa_http_endpoints._state.initialize(settings)
        iisa_http_endpoints._state.refresh_data()

        client = TestClient(app, raise_server_exceptions=False)

        # Act
        response = client.post("/get-score", json={"indexer_id": "0xNONEXISTENT"})

        # Assert
        assert response.status_code == 200
        data = response.json()
        assert data["indexer_id"] == "0xNONEXISTENT"
        assert data["found"] is False
        assert data["weighted_score"] is None
        assert data["components"] is None

    def test_get_score_no_data(self):
        """Verify 503 when data not loaded."""
        # Arrange
        from iisa import iisa_http_endpoints
        from iisa.iisa_http_endpoints import app

        iisa_http_endpoints._state._history = None

        client = TestClient(app, raise_server_exceptions=False)

        # Act
        response = client.post("/get-score", json={"indexer_id": "A"})

        # Assert
        assert response.status_code == 503


class TestSelectIndexersEndpoint:
    """Tests for POST /select-indexers endpoint."""

    @patch("iisa.iisa_http_endpoints.IndexerSelector")
    def test_select_indexers_returns_deployment_id_and_indexers(
        self, mock_processor_class, mock_history_df
    ):
        """With data loaded, mock IndexerSelector, verify response structure."""
        # Arrange
        from iisa import iisa_http_endpoints
        from iisa.iisa_http_endpoints import app

        mock_processor = MagicMock()
        mock_processor.current_group = ["0xABC", "0xXYZ", "0x123"]
        mock_processor_class.return_value = mock_processor

        iisa_http_endpoints._state._history = mock_history_df
        iisa_http_endpoints._state._initialized = True

        client = TestClient(app, raise_server_exceptions=False)

        # Act
        response = client.post(
            "/select-indexers",
            json={
                "deployment_id": "Qm123",
                "num_candidates": 3,
            },
        )

        # Assert
        assert response.status_code == 200
        data = response.json()
        assert data["deployment_id"] == "Qm123"
        assert len(data["indexers"]) == 3
        indexer_ids = [i["id"] for i in data["indexers"]]
        assert indexer_ids == ["0xABC", "0xXYZ", "0x123"]

    def test_select_indexers_no_data_returns_503(self):
        """Without data loaded, verify 503 returned."""
        # Arrange
        from iisa import iisa_http_endpoints
        from iisa.iisa_http_endpoints import app

        iisa_http_endpoints._state._history = None
        iisa_http_endpoints._state._initialized = False

        client = TestClient(app, raise_server_exceptions=False)

        # Act
        response = client.post(
            "/select-indexers",
            json={
                "deployment_id": "Qm123",
                "num_candidates": 3,
            },
        )

        # Assert
        assert response.status_code == 503
        assert "IISA data not loaded" in response.json()["detail"]

    @patch("iisa.iisa_http_endpoints.IndexerSelector")
    def test_select_indexers_processor_exception_returns_500(
        self, mock_processor_class, mock_history_df
    ):
        """IndexerSelector raises, verify 500 returned."""
        # Arrange
        from iisa import iisa_http_endpoints
        from iisa.iisa_http_endpoints import app

        mock_processor_class.side_effect = Exception("Processing failed")

        iisa_http_endpoints._state._history = mock_history_df
        iisa_http_endpoints._state._initialized = True

        client = TestClient(app, raise_server_exceptions=False)

        # Act
        response = client.post(
            "/select-indexers",
            json={
                "deployment_id": "Qm123",
                "num_candidates": 3,
            },
        )

        # Assert
        assert response.status_code == 500
        assert "Selection failed: Processing failed" in response.json()["detail"]

    @patch("iisa.iisa_http_endpoints.IndexerSelector")
    def test_select_indexers_zero_num_candidates(self, mock_processor_class, mock_history_df):
        """Verify empty list returned when num_candidates is 0."""
        # Arrange
        from iisa import iisa_http_endpoints
        from iisa.iisa_http_endpoints import app

        iisa_http_endpoints._state._history = mock_history_df
        iisa_http_endpoints._state._initialized = True

        client = TestClient(app, raise_server_exceptions=False)

        # Act
        response = client.post(
            "/select-indexers",
            json={
                "deployment_id": "Qm123",
                "num_candidates": 0,
            },
        )

        # Assert
        assert response.status_code == 200
        data = response.json()
        assert data["deployment_id"] == "Qm123"
        assert len(data["indexers"]) == 0
        mock_processor_class.assert_not_called()

    @patch("iisa.iisa_http_endpoints.IndexerSelector")
    def test_select_indexers_empty_result(self, mock_processor_class, mock_history_df):
        """IndexerSelector returns no selection, verify empty indexers list."""
        # Arrange
        from iisa import iisa_http_endpoints
        from iisa.iisa_http_endpoints import app

        mock_processor = MagicMock()
        mock_processor.current_group = []
        mock_processor_class.return_value = mock_processor

        iisa_http_endpoints._state._history = mock_history_df
        iisa_http_endpoints._state._initialized = True

        client = TestClient(app, raise_server_exceptions=False)

        # Act
        response = client.post(
            "/select-indexers",
            json={
                "deployment_id": "Qm123",
                "num_candidates": 3,
            },
        )

        # Assert
        assert response.status_code == 200
        data = response.json()
        assert data["deployment_id"] == "Qm123"
        assert len(data["indexers"]) == 0

    @patch("iisa.iisa_http_endpoints.IndexerSelector")
    def test_select_indexers_passes_target_size(self, mock_processor_class, mock_history_df):
        """Verify num_candidates passed as target_size to IndexerSelector."""
        # Arrange
        from iisa import iisa_http_endpoints
        from iisa.iisa_http_endpoints import app

        mock_processor = MagicMock()
        mock_processor.current_group = ["0xabc"]
        mock_processor_class.return_value = mock_processor

        iisa_http_endpoints._state._history = mock_history_df
        iisa_http_endpoints._state._initialized = True

        client = TestClient(app, raise_server_exceptions=False)

        # Act
        client.post(
            "/select-indexers",
            json={
                "deployment_id": "Qm123",
                "num_candidates": 5,
            },
        )

        # Assert
        call_kwargs = mock_processor_class.call_args[1]
        assert call_kwargs["target_size"] == 5


class TestSelectWithProcessor:
    """Tests for _select_with_processor helper."""

    @patch("iisa.iisa_http_endpoints.IndexerSelector")
    def test_select_with_processor_returns_selection_response(
        self, mock_processor_class, mock_history_df
    ):
        """Mock IndexerSelector.current_group, verify SelectionResponse returned."""
        # Arrange
        from iisa import iisa_http_endpoints
        from iisa.iisa_http_endpoints import (
            SelectionRequest,
            SelectionResponse,
            _select_with_processor,
        )

        mock_processor = MagicMock()
        mock_processor.current_group = ["0xABC", "0xXYZ", "0x123"]
        mock_processor_class.return_value = mock_processor

        iisa_http_endpoints._state._history = mock_history_df

        request = SelectionRequest(
            deployment_id="Qm123",
            existing_indexers=["0xEXIST"],
            num_candidates=3,
        )

        # Act
        result = _select_with_processor(request)

        # Assert
        assert isinstance(result, SelectionResponse)
        assert result.deployment_id == "Qm123"
        assert len(result.indexers) == 3
        assert [i.id for i in result.indexers] == ["0xABC", "0xXYZ", "0x123"]
        mock_processor_class.assert_called_once()

    def test_select_with_processor_no_history(self):
        """_state.history=None, verify empty SelectionResponse."""
        # Arrange
        from iisa import iisa_http_endpoints
        from iisa.iisa_http_endpoints import SelectionRequest, _select_with_processor

        iisa_http_endpoints._state._history = None

        request = SelectionRequest(deployment_id="Qm123", num_candidates=3)

        # Act
        result = _select_with_processor(request)

        # Assert
        assert result.deployment_id == "Qm123"
        assert result.indexers == []

    @patch("iisa.iisa_http_endpoints.IndexerSelector")
    def test_select_with_processor_maps_blocklist(self, mock_processor_class, mock_history_df):
        """Verify blocklist mapped to indexer_denylist param."""
        # Arrange
        from iisa import iisa_http_endpoints
        from iisa.iisa_http_endpoints import SelectionRequest, _select_with_processor

        mock_processor = MagicMock()
        mock_processor.current_group = []
        mock_processor_class.return_value = mock_processor

        iisa_http_endpoints._state._history = mock_history_df

        request = SelectionRequest(
            deployment_id="Qm123",
            blocklist=["0xBAD1", "0xBAD2"],
            num_candidates=3,
        )

        # Act
        _select_with_processor(request)

        # Assert
        call_kwargs = mock_processor_class.call_args[1]
        assert call_kwargs["indexer_denylist"] == ["0xBAD1", "0xBAD2"]

    @patch("iisa.iisa_http_endpoints.IndexerSelector")
    def test_select_with_processor_builds_existing_agreements(
        self, mock_processor_class, mock_history_df
    ):
        """Verify existing_indexers mapped to existing_agreements dict."""
        # Arrange
        from iisa import iisa_http_endpoints
        from iisa.iisa_http_endpoints import SelectionRequest, _select_with_processor

        mock_processor = MagicMock()
        mock_processor.current_group = []
        mock_processor_class.return_value = mock_processor

        iisa_http_endpoints._state._history = mock_history_df

        request = SelectionRequest(
            deployment_id="Qm123",
            existing_indexers=["0xEXIST1", "0xEXIST2"],
            num_candidates=3,
        )

        # Act
        _select_with_processor(request)

        # Assert
        call_kwargs = mock_processor_class.call_args[1]
        assert call_kwargs["existing_agreements"] == {"Qm123": ["0xEXIST1", "0xEXIST2"]}

    @patch("iisa.iisa_http_endpoints.IndexerSelector")
    def test_select_with_processor_passes_pending_agreements(
        self, mock_processor_class, mock_history_df
    ):
        """Verify pending_agreements passed through."""
        # Arrange
        from iisa import iisa_http_endpoints
        from iisa.iisa_http_endpoints import SelectionRequest, _select_with_processor

        mock_processor = MagicMock()
        mock_processor.current_group = []
        mock_processor_class.return_value = mock_processor

        iisa_http_endpoints._state._history = mock_history_df

        pending = {"Qm123": ["0xPEND1"], "Qm456": ["0xPEND2"]}
        request = SelectionRequest(
            deployment_id="Qm123",
            pending_agreements=pending,
            num_candidates=3,
        )

        # Act
        _select_with_processor(request)

        # Assert
        call_kwargs = mock_processor_class.call_args[1]
        assert call_kwargs["pending_agreements"] == pending

    @patch("iisa.iisa_http_endpoints.IndexerSelector")
    def test_select_with_processor_passes_declined_indexers(
        self, mock_processor_class, mock_history_df
    ):
        """Verify declined_indexers passed through."""
        # Arrange
        from iisa import iisa_http_endpoints
        from iisa.iisa_http_endpoints import SelectionRequest, _select_with_processor

        mock_processor = MagicMock()
        mock_processor.current_group = []
        mock_processor_class.return_value = mock_processor

        iisa_http_endpoints._state._history = mock_history_df

        declined = {"Qm123": ["0xDEC1", "0xDEC2"]}
        request = SelectionRequest(
            deployment_id="Qm123",
            declined_indexers=declined,
            num_candidates=3,
        )

        # Act
        _select_with_processor(request)

        # Assert
        call_kwargs = mock_processor_class.call_args[1]
        assert call_kwargs["declined_indexers"] == declined

    @patch("iisa.iisa_http_endpoints.IndexerSelector")
    def test_select_with_processor_passes_target_size(self, mock_processor_class, mock_history_df):
        """Verify num_candidates passed as target_size to IndexerSelector."""
        # Arrange
        from iisa import iisa_http_endpoints
        from iisa.iisa_http_endpoints import SelectionRequest, _select_with_processor

        mock_processor = MagicMock()
        mock_processor.current_group = []
        mock_processor_class.return_value = mock_processor

        iisa_http_endpoints._state._history = mock_history_df

        request = SelectionRequest(
            deployment_id="Qm123",
            num_candidates=5,
        )

        # Act
        _select_with_processor(request)

        # Assert - target_size
        call_kwargs = mock_processor_class.call_args[1]
        assert call_kwargs["target_size"] == 5

    @patch("iisa.iisa_http_endpoints.IndexerSelector")
    def test_select_with_processor_passes_synced_indexers(
        self, mock_processor_class, mock_history_df
    ):
        """Verify synced indexers from sync status threaded to IndexerSelector."""
        from iisa import iisa_http_endpoints
        from iisa.iisa_http_endpoints import SelectionRequest, _select_with_processor

        mock_processor = MagicMock()
        mock_processor.current_group = []
        mock_processor_class.return_value = mock_processor

        iisa_http_endpoints._state._history = mock_history_df

        # Set up sync status with known synced indexers
        mock_sync = MagicMock()
        mock_sync.synced_indexers_for.return_value = {"0xaaa", "0xbbb"}
        iisa_http_endpoints._state._sync_status = mock_sync

        request = SelectionRequest(
            deployment_id="Qm123",
            num_candidates=3,
        )

        # Act
        _select_with_processor(request)

        # Assert
        call_kwargs = mock_processor_class.call_args[1]
        assert call_kwargs["synced_indexers"] == {"0xaaa", "0xbbb"}

        # Cleanup
        iisa_http_endpoints._state._sync_status = None

    @patch("iisa.iisa_http_endpoints.IndexerSelector")
    def test_select_with_processor_no_sync_status(self, mock_processor_class, mock_history_df):
        """Without sync status, synced_indexers is empty set."""
        from iisa import iisa_http_endpoints
        from iisa.iisa_http_endpoints import SelectionRequest, _select_with_processor

        mock_processor = MagicMock()
        mock_processor.current_group = []
        mock_processor_class.return_value = mock_processor

        iisa_http_endpoints._state._history = mock_history_df
        iisa_http_endpoints._state._sync_status = None

        request = SelectionRequest(
            deployment_id="Qm123",
            num_candidates=3,
        )

        # Act
        _select_with_processor(request)

        # Assert
        call_kwargs = mock_processor_class.call_args[1]
        assert call_kwargs["synced_indexers"] == set()


class TestPushSyncStatusEndpoint:
    """Tests for POST /sync-status endpoint (fetcher → iisa push)."""

    @staticmethod
    def _sample_payload():
        return {
            "0xAAA": {
                "deployments": ["QmDeploy1"],
                "fetched_at": datetime.now(timezone.utc).isoformat(),
            }
        }

    def test_push_sync_status_success(self, tmp_path, monkeypatch):
        from iisa import iisa_http_endpoints
        from iisa.iisa_http_endpoints import Settings, app

        iisa_http_endpoints.get_settings.cache_clear()
        settings = Settings()
        # Redirect the sync-status cache path into tmp_path
        settings.sync_status_file_path = str(tmp_path / "sync_status.json")
        iisa_http_endpoints._state.initialize(settings)

        client = TestClient(app, raise_server_exceptions=False)
        response = client.post("/sync-status", json=self._sample_payload())

        assert response.status_code == 200
        body = response.json()
        assert body["status"] == "success"
        assert body["indexers"] == 1
        assert (tmp_path / "sync_status.json").exists()

    def test_push_sync_status_empty_object_accepted(self, tmp_path, monkeypatch):
        """Empty dict payload (no indexers synced) should be accepted."""
        from iisa import iisa_http_endpoints
        from iisa.iisa_http_endpoints import Settings, app

        iisa_http_endpoints.get_settings.cache_clear()
        settings = Settings()
        settings.sync_status_file_path = str(tmp_path / "sync_status.json")
        iisa_http_endpoints._state.initialize(settings)

        client = TestClient(app, raise_server_exceptions=False)
        response = client.post("/sync-status", json={})
        assert response.status_code == 200
        assert response.json()["indexers"] == 0

    def test_push_sync_status_rejects_wrong_token(self, monkeypatch):
        monkeypatch.setenv("IISA_PUSH_TOKEN", "secret")
        from iisa import iisa_http_endpoints
        from iisa.iisa_http_endpoints import Settings, app

        iisa_http_endpoints.get_settings.cache_clear()
        iisa_http_endpoints._state.initialize(Settings())

        client = TestClient(app, raise_server_exceptions=False)
        response = client.post(
            "/sync-status",
            json=self._sample_payload(),
            headers={"Authorization": "Bearer wrong"},
        )
        assert response.status_code == 401


class TestRequirePushTokenStartup:
    """Tests for the IISA_REQUIRE_PUSH_TOKEN hard-fail gate on lifespan."""

    def test_startup_fails_when_required_and_token_missing(self, monkeypatch):
        """IISA_REQUIRE_PUSH_TOKEN=true + unset token → RuntimeError at startup."""
        monkeypatch.setenv("IISA_REQUIRE_PUSH_TOKEN", "true")
        monkeypatch.delenv("IISA_PUSH_TOKEN", raising=False)

        from iisa import iisa_http_endpoints
        from iisa.iisa_http_endpoints import app

        iisa_http_endpoints.get_settings.cache_clear()

        # TestClient's context manager runs lifespan on __enter__. The
        # RuntimeError from the gate surfaces there.
        with pytest.raises(RuntimeError, match="IISA_PUSH_TOKEN is required"):
            with TestClient(app) as _client:
                pass

    @patch("iisa.iisa_http_endpoints.DataManager")
    @patch("iisa.iisa_http_endpoints.FileScoreLoader")
    def test_startup_succeeds_when_required_and_token_set(
        self, mock_loader_class, mock_dm_class, monkeypatch
    ):
        """IISA_REQUIRE_PUSH_TOKEN=true + token set → service starts normally."""
        monkeypatch.setenv("IISA_REQUIRE_PUSH_TOKEN", "true")
        monkeypatch.setenv("IISA_PUSH_TOKEN", "secret")

        from iisa import iisa_http_endpoints
        from iisa.iisa_http_endpoints import app

        iisa_http_endpoints.get_settings.cache_clear()

        mock_dm_instance = MagicMock()
        mock_dm_instance.load_scores.return_value = False  # empty cache is fine
        mock_dm_class.return_value = mock_dm_instance

        with TestClient(app) as client:
            response = client.get("/health")
            assert response.status_code == 200

    @patch("iisa.iisa_http_endpoints.DataManager")
    @patch("iisa.iisa_http_endpoints.FileScoreLoader")
    def test_startup_warns_when_not_required_and_token_missing(
        self, mock_loader_class, mock_dm_class, monkeypatch, caplog
    ):
        """No require flag, no token → startup WARNING, service accepts requests."""
        import logging

        monkeypatch.delenv("IISA_REQUIRE_PUSH_TOKEN", raising=False)
        monkeypatch.delenv("IISA_PUSH_TOKEN", raising=False)

        from iisa import iisa_http_endpoints
        from iisa.iisa_http_endpoints import app

        iisa_http_endpoints.get_settings.cache_clear()

        mock_dm_instance = MagicMock()
        mock_dm_instance.load_scores.return_value = False
        mock_dm_class.return_value = mock_dm_instance

        with caplog.at_level(logging.WARNING, logger="iisa-service"):
            with TestClient(app) as client:
                response = client.get("/health")
                assert response.status_code == 200

        assert any("IISA_PUSH_TOKEN is not set" in rec.message for rec in caplog.records)


class TestExtractChainPrice:
    """Tests for _extract_chain_price -- parses per-chain price from JSON blob."""

    def test_extracts_price_for_matching_chain(self):
        from iisa.iisa_http_endpoints import _extract_chain_price

        prices_json = json.dumps({"arbitrum-one": 450.0, "mainnet": 200.0})
        assert _extract_chain_price(prices_json, "arbitrum-one") == 450.0
        assert _extract_chain_price(prices_json, "mainnet") == 200.0

    def test_returns_none_for_missing_chain(self):
        from iisa.iisa_http_endpoints import _extract_chain_price

        prices_json = json.dumps({"arbitrum-one": 450.0})
        assert _extract_chain_price(prices_json, "optimism") is None

    def test_returns_none_for_empty_json(self):
        from iisa.iisa_http_endpoints import _extract_chain_price

        assert _extract_chain_price("{}", "arbitrum-one") is None

    def test_returns_none_for_malformed_json(self):
        from iisa.iisa_http_endpoints import _extract_chain_price

        assert _extract_chain_price("not json", "arbitrum-one") is None

    def test_returns_none_for_none_input(self):
        from iisa.iisa_http_endpoints import _extract_chain_price

        assert _extract_chain_price(None, "arbitrum-one") is None

    def test_converts_string_price_to_float(self):
        from iisa.iisa_http_endpoints import _extract_chain_price

        prices_json = json.dumps({"arbitrum-one": "450.5"})
        assert _extract_chain_price(prices_json, "arbitrum-one") == 450.5


class TestFilterByPrice:
    """Tests for _filter_by_price -- excludes indexers by DIP info, chain, and budget."""

    def _make_history(self, rows):
        return pd.DataFrame(rows)

    def test_no_filtering_when_chain_id_is_none(self):
        from iisa.iisa_http_endpoints import _filter_by_price

        df = self._make_history([{"indexer": "0xA", "dips_info_available": False}])
        result, reason = _filter_by_price(df, None, None)
        assert len(result) == 1

    def test_excludes_indexers_without_dips_info(self):
        from iisa.iisa_http_endpoints import _filter_by_price

        df = self._make_history(
            [
                {
                    "indexer": "0xA",
                    "dips_info_available": True,
                    "dips_supported_networks": json.dumps(["arb"]),
                    "dips_min_grt_per_30_days": json.dumps({"arb": 100}),
                },
                {
                    "indexer": "0xB",
                    "dips_info_available": False,
                    "dips_supported_networks": json.dumps(["arb"]),
                    "dips_min_grt_per_30_days": json.dumps({"arb": 100}),
                },
            ]
        )
        result, reason = _filter_by_price(df, "arb", None)
        assert len(result) == 1
        assert result.iloc[0]["indexer"] == "0xA"

    def test_all_lacking_dips_info_returns_empty_with_reason(self):
        from iisa.iisa_http_endpoints import _filter_by_price

        df = self._make_history(
            [
                {"indexer": "0xA", "dips_info_available": False},
                {"indexer": "0xB", "dips_info_available": False},
            ]
        )
        result, reason = _filter_by_price(df, "arb", None)
        assert result.empty
        assert "lack DIP info" in reason

    def test_excludes_indexers_not_supporting_chain(self):
        from iisa.iisa_http_endpoints import _filter_by_price

        df = self._make_history(
            [
                {
                    "indexer": "0xA",
                    "dips_info_available": True,
                    "dips_supported_networks": json.dumps(["arbitrum-one"]),
                    "dips_min_grt_per_30_days": json.dumps({"arbitrum-one": 100}),
                },
                {
                    "indexer": "0xB",
                    "dips_info_available": True,
                    "dips_supported_networks": json.dumps(["mainnet"]),
                    "dips_min_grt_per_30_days": json.dumps({"mainnet": 200}),
                },
            ]
        )
        result, reason = _filter_by_price(df, "arbitrum-one", None)
        assert len(result) == 1
        assert result.iloc[0]["indexer"] == "0xA"

    def test_excludes_indexers_over_budget(self):
        from iisa.iisa_http_endpoints import _filter_by_price

        df = self._make_history(
            [
                {
                    "indexer": "0xA",
                    "dips_info_available": True,
                    "dips_supported_networks": json.dumps(["arb"]),
                    "dips_min_grt_per_30_days": json.dumps({"arb": 100}),
                },
                {
                    "indexer": "0xB",
                    "dips_info_available": True,
                    "dips_supported_networks": json.dumps(["arb"]),
                    "dips_min_grt_per_30_days": json.dumps({"arb": 500}),
                },
            ]
        )
        result, reason = _filter_by_price(df, "arb", 200.0)
        assert len(result) == 1
        assert result.iloc[0]["indexer"] == "0xA"

    def test_all_over_budget_returns_empty_with_reason(self):
        from iisa.iisa_http_endpoints import _filter_by_price

        df = self._make_history(
            [
                {
                    "indexer": "0xA",
                    "dips_info_available": True,
                    "dips_supported_networks": json.dumps(["arb"]),
                    "dips_min_grt_per_30_days": json.dumps({"arb": 500}),
                },
            ]
        )
        result, reason = _filter_by_price(df, "arb", 200.0)
        assert result.empty
        assert "exceed payment ceiling" in reason

    def test_indexer_without_chain_pricing_excluded(self):
        from iisa.iisa_http_endpoints import _filter_by_price

        df = self._make_history(
            [
                {
                    "indexer": "0xA",
                    "dips_info_available": True,
                    "dips_supported_networks": json.dumps(["arb"]),
                    "dips_min_grt_per_30_days": json.dumps({"mainnet": 100}),
                },
            ]
        )
        result, reason = _filter_by_price(df, "arb", None)
        assert result.empty


class TestBuildSelectedIndexers:
    """Tests for _build_selected_indexers -- extracts chain-specific price into response."""

    def test_returns_chain_specific_price(self):
        from iisa.iisa_http_endpoints import _build_selected_indexers

        history = pd.DataFrame(
            [
                {
                    "indexer": "0xa",
                    "dips_min_grt_per_30_days": json.dumps(
                        {"arbitrum-one": 450.0, "mainnet": 200.0}
                    ),
                    "dips_min_grt_per_billion_entities_per_30_days": 2000.0,
                }
            ]
        )
        result = _build_selected_indexers(["0xa"], history, "arbitrum-one")
        assert len(result) == 1
        assert result[0].min_grt_per_30_days == 450.0
        assert result[0].min_grt_per_billion_entities_per_30_days == 2000.0

    def test_returns_none_when_chain_not_in_pricing(self):
        from iisa.iisa_http_endpoints import _build_selected_indexers

        history = pd.DataFrame(
            [
                {
                    "indexer": "0xa",
                    "dips_min_grt_per_30_days": json.dumps({"mainnet": 200.0}),
                    "dips_min_grt_per_billion_entities_per_30_days": None,
                }
            ]
        )
        result = _build_selected_indexers(["0xa"], history, "arbitrum-one")
        assert result[0].min_grt_per_30_days is None

    def test_returns_none_when_no_chain_id(self):
        from iisa.iisa_http_endpoints import _build_selected_indexers

        history = pd.DataFrame(
            [
                {
                    "indexer": "0xa",
                    "dips_min_grt_per_30_days": json.dumps({"arbitrum-one": 450.0}),
                }
            ]
        )
        result = _build_selected_indexers(["0xa"], history, None)
        assert result[0].min_grt_per_30_days is None

    def test_indexer_not_in_history_returns_none_pricing(self):
        from iisa.iisa_http_endpoints import _build_selected_indexers

        history = pd.DataFrame([{"indexer": "0xother"}])
        result = _build_selected_indexers(["0xa"], history, "arbitrum-one")
        assert result[0].min_grt_per_30_days is None


class TestEnrichWithChainPrices:
    """Tests for _enrich_with_chain_prices -- adds price columns for scoring."""

    def test_adds_chain_specific_base_price(self):
        from iisa.iisa_http_endpoints import _enrich_with_chain_prices

        df = pd.DataFrame(
            [
                {
                    "indexer": "0xa",
                    "dips_min_grt_per_30_days": json.dumps({"arb": 450.0}),
                    "dips_min_grt_per_billion_entities_per_30_days": 2000.0,
                }
            ]
        )
        result = _enrich_with_chain_prices(df, "arb")
        assert result.iloc[0]["base_price_per_epoch"] == 450.0
        assert result.iloc[0]["price_per_entity"] == 2000.0

    def test_zero_price_when_chain_not_found(self):
        from iisa.iisa_http_endpoints import _enrich_with_chain_prices

        df = pd.DataFrame(
            [
                {
                    "indexer": "0xa",
                    "dips_min_grt_per_30_days": json.dumps({"mainnet": 200.0}),
                }
            ]
        )
        result = _enrich_with_chain_prices(df, "arb")
        assert result.iloc[0]["base_price_per_epoch"] == 0.0

    def test_zero_price_when_no_chain_id(self):
        from iisa.iisa_http_endpoints import _enrich_with_chain_prices

        df = pd.DataFrame(
            [
                {
                    "indexer": "0xa",
                    "dips_min_grt_per_30_days": json.dumps({"arb": 450.0}),
                }
            ]
        )
        result = _enrich_with_chain_prices(df, None)
        assert result.iloc[0]["base_price_per_epoch"] == 0.0


class TestSelectIndexersEndToEndPricing:
    """Integration test: /select-indexers must not return indexers with None pricing."""

    def test_excludes_indexers_without_dips_info_from_response(self):
        """Indexers without DIP info or without pricing for the requested chain
        must not appear in the /select-indexers response. If they did, dipper
        would fall back to its static pricing_table (potentially 10x the market
        rate) with no signal."""
        from iisa import iisa_http_endpoints
        from iisa.iisa_http_endpoints import app

        # 3 indexers:
        #   0xa - has DIP info and pricing for "arb" (should be selected)
        #   0xb - has DIP info but pricing only for "mainnet" (wrong chain)
        #   0xc - dips_info_available = False (no DIP info at all)
        history = pd.DataFrame(
            [
                {
                    "indexer": "0xa",
                    "dips_info_available": True,
                    "dips_supported_networks": json.dumps(["arb"]),
                    "dips_min_grt_per_30_days": json.dumps({"arb": 450.0}),
                    "dips_min_grt_per_billion_entities_per_30_days": 2000.0,
                },
                {
                    "indexer": "0xb",
                    "dips_info_available": True,
                    "dips_supported_networks": json.dumps(["mainnet"]),
                    "dips_min_grt_per_30_days": json.dumps({"mainnet": 200.0}),
                    "dips_min_grt_per_billion_entities_per_30_days": 1000.0,
                },
                {
                    "indexer": "0xc",
                    "dips_info_available": False,
                    "dips_supported_networks": json.dumps(["arb"]),
                    "dips_min_grt_per_30_days": json.dumps({"arb": 100.0}),
                    "dips_min_grt_per_billion_entities_per_30_days": 500.0,
                },
            ]
        )

        iisa_http_endpoints._state._history = history
        iisa_http_endpoints._state._initialized = True

        client = TestClient(app, raise_server_exceptions=False)

        response = client.post(
            "/select-indexers",
            json={
                "deployment_id": "QmTest123",
                "chain_id": "arb",
                "num_candidates": 3,
            },
        )

        assert response.status_code == 200
        data = response.json()
        indexers = data["indexers"]

        # Only 0xa should be returned
        indexer_ids = [i["id"] for i in indexers]
        assert "0xa" in indexer_ids
        assert "0xb" not in indexer_ids, (
            "indexer without pricing for requested chain should be excluded"
        )
        assert "0xc" not in indexer_ids, "indexer without DIP info should be excluded"

        # The returned indexer must have a non-None price
        for indexer in indexers:
            assert indexer["min_grt_per_30_days"] is not None, (
                f"indexer {indexer['id']} returned with None pricing -- "
                "dipper would fall back to static pricing_table"
            )

    @staticmethod
    def _row(address, chain, price, loc="US", org="org1"):
        """Build a minimal indexer row with all columns IndexerSelector needs."""
        return {
            "indexer": address,
            "dips_info_available": True,
            "dips_supported_networks": json.dumps([chain]),
            "dips_min_grt_per_30_days": json.dumps({chain: price}),
            "dips_min_grt_per_billion_entities_per_30_days": 500.0,
            "destination_loc": loc,
            "org": org,
        }

    def test_budget_enforcement_excludes_expensive_indexers(self):
        """Indexers whose price exceeds max_grt_per_30_days must be excluded."""
        from iisa import iisa_http_endpoints
        from iisa.iisa_http_endpoints import app

        history = pd.DataFrame(
            [
                self._row("0xcheap", "arb", 100.0),
                self._row("0xmid", "arb", 300.0),
                self._row("0xexpensive", "arb", 5000.0),
            ]
        )

        iisa_http_endpoints._state._history = history
        iisa_http_endpoints._state._initialized = True

        client = TestClient(app, raise_server_exceptions=False)

        response = client.post(
            "/select-indexers",
            json={
                "deployment_id": "QmBudgetTest",
                "chain_id": "arb",
                "num_candidates": 3,
                "max_grt_per_30_days": 400.0,
            },
        )

        assert response.status_code == 200
        indexer_ids = [i["id"] for i in response.json()["indexers"]]
        assert "0xcheap" in indexer_ids
        assert "0xmid" in indexer_ids
        assert "0xexpensive" not in indexer_ids, (
            "indexer priced at 5000 GRT/30d should be excluded with budget of 400"
        )

    def test_blocklist_excludes_indexers(self):
        """Blocklisted indexers must not appear in the response."""
        from iisa import iisa_http_endpoints
        from iisa.iisa_http_endpoints import app

        history = pd.DataFrame(
            [
                self._row("0xgood", "arb", 100.0),
                self._row("0xblocked", "arb", 100.0),
            ]
        )

        iisa_http_endpoints._state._history = history
        iisa_http_endpoints._state._initialized = True

        client = TestClient(app, raise_server_exceptions=False)

        response = client.post(
            "/select-indexers",
            json={
                "deployment_id": "QmBlocklistTest",
                "chain_id": "arb",
                "num_candidates": 2,
                "blocklist": ["0xblocked"],
            },
        )

        assert response.status_code == 200
        indexer_ids = [i["id"] for i in response.json()["indexers"]]
        assert "0xgood" in indexer_ids
        assert "0xblocked" not in indexer_ids, "blocklisted indexer should be excluded"

    def test_num_candidates_caps_response_size(self):
        """Response must not contain more indexers than num_candidates."""
        from iisa import iisa_http_endpoints
        from iisa.iisa_http_endpoints import app

        history = pd.DataFrame(
            [
                self._row(f"0x{i:040x}", "arb", 100.0 + i, loc=f"loc{i}", org=f"org{i}")
                for i in range(5)
            ]
        )

        iisa_http_endpoints._state._history = history
        iisa_http_endpoints._state._initialized = True

        client = TestClient(app, raise_server_exceptions=False)

        response = client.post(
            "/select-indexers",
            json={
                "deployment_id": "QmCapTest",
                "chain_id": "arb",
                "num_candidates": 2,
            },
        )

        assert response.status_code == 200
        indexers = response.json()["indexers"]
        assert len(indexers) <= 2, f"requested 2 candidates but got {len(indexers)}"
