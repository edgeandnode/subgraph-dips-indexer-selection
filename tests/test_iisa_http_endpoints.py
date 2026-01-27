"""Tests for the IISA HTTP API endpoints."""

import os
from functools import lru_cache
from unittest.mock import MagicMock, patch

import pandas as pd
import pytest
from fastapi.testclient import TestClient
from pydantic import ValidationError


class TestSettings:
    """Tests for Settings class and get_settings()."""

    def test_settings_loads_from_env(self):
        """Verify IISA_ prefix environment variables load correctly."""
        # Arrange
        env_vars = {
            "IISA_GCP_PROJECT": "test-project",
            "IISA_GCP_LOCATION": "EU",
            "IISA_HOST": "127.0.0.1",
            "IISA_PORT": "9000",
            "IISA_LOG_LEVEL": "DEBUG",
        }

        # Act
        with patch.dict(os.environ, env_vars, clear=False):
            from iisa.iisa_http_endpoints import Settings

            settings = Settings()

        # Assert
        assert settings.gcp_project == "test-project"
        assert settings.gcp_location == "EU"
        assert settings.host == "127.0.0.1"
        assert settings.port == 9000
        assert settings.log_level == "DEBUG"

    def test_settings_default_values(self):
        """Verify defaults: gcp_location="US", host="0.0.0.0", port=8080, log_level="INFO"."""
        # Arrange
        env_vars = {"IISA_GCP_PROJECT": "test-project"}

        # Act
        with patch.dict(os.environ, env_vars, clear=False):
            from iisa.iisa_http_endpoints import Settings

            settings = Settings()

        # Assert
        assert settings.gcp_location == "US"
        assert settings.host == "0.0.0.0"
        assert settings.port == 8080
        assert settings.log_level == "INFO"

    def test_settings_requires_gcp_project(self):
        """Verify ValidationError when IISA_GCP_PROJECT missing."""
        # Arrange - remove the required env var if present
        env_without_project = {k: v for k, v in os.environ.items() if k != "IISA_GCP_PROJECT"}

        # Act & Assert
        with patch.dict(os.environ, env_without_project, clear=True):
            from iisa.iisa_http_endpoints import Settings

            with pytest.raises(ValidationError) as exc_info:
                Settings()

            assert "gcp_project" in str(exc_info.value)

    def test_get_settings_cached(self):
        """Verify @lru_cache returns same instance."""
        # Arrange
        env_vars = {"IISA_GCP_PROJECT": "test-project"}

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

    def test_candidate_indexer_valid(self):
        """Valid id and url fields."""
        # Arrange & Act
        from iisa.iisa_http_endpoints import CandidateIndexer

        candidate = CandidateIndexer(id="0xABC123", url="https://indexer.example.com/")

        # Assert
        assert candidate.id == "0xABC123"
        assert candidate.url == "https://indexer.example.com/"

    def test_selection_request_all_optional(self):
        """All fields optional except deployment_id."""
        # Arrange & Act
        from iisa.iisa_http_endpoints import SelectionRequest

        request = SelectionRequest(deployment_id="Qm123")

        # Assert
        assert request.deployment_id == "Qm123"
        assert request.candidates is None
        assert request.existing_indexers is None
        assert request.pending_agreements is None
        assert request.num_candidates is None
        assert request.blocklist is None
        assert request.declined_indexers is None

    def test_selection_request_with_candidates(self):
        """Verify candidates list serialization."""
        # Arrange & Act
        from iisa.iisa_http_endpoints import CandidateIndexer, SelectionRequest

        candidates = [
            CandidateIndexer(id="0xABC", url="https://a.com/"),
            CandidateIndexer(id="0xXYZ", url="https://b.com/"),
        ]
        request = SelectionRequest(
            deployment_id="Qm123",
            candidates=candidates,
            existing_indexers=["0x111"],
            num_candidates=2,
            blocklist=["0xBAD"],
        )

        # Assert
        assert len(request.candidates) == 2
        assert request.candidates[0].id == "0xABC"
        assert request.existing_indexers == ["0x111"]
        assert request.blocklist == ["0xBAD"]

    def test_single_selection_response(self):
        """Optional indexer_id field."""
        # Arrange & Act
        from iisa.iisa_http_endpoints import SingleSelectionResponse

        response_with_id = SingleSelectionResponse(indexer_id="0xABC")
        response_without_id = SingleSelectionResponse()

        # Assert
        assert response_with_id.indexer_id == "0xABC"
        assert response_without_id.indexer_id is None

    def test_multi_selection_response(self):
        """indexer_ids list field."""
        # Arrange & Act
        from iisa.iisa_http_endpoints import MultiSelectionResponse

        response = MultiSelectionResponse(indexer_ids=["0xABC", "0xXYZ", "0x123"])

        # Assert
        assert response.indexer_ids == ["0xABC", "0xXYZ", "0x123"]
        assert len(response.indexer_ids) == 3

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
        """Verify initial state: settings=None, data_manager=None, _history=None, _initialized=False."""
        # Arrange & Act
        from iisa.iisa_http_endpoints import IISAState

        state = IISAState()

        # Assert
        assert state.settings is None
        assert state.data_manager is None
        assert state._history is None
        assert state._initialized is False

    @patch("iisa.iisa_http_endpoints.DataManager")
    @patch("iisa.iisa_http_endpoints.BigQueryProvider")
    def test_initialize_success(self, mock_bq_class, mock_dm_class):
        """Mock BigQueryProvider/DataManager, verify _initialized=True."""
        # Arrange
        from iisa.iisa_http_endpoints import IISAState, Settings

        mock_bq_instance = MagicMock()
        mock_bq_class.return_value = mock_bq_instance

        mock_dm_instance = MagicMock()
        mock_dm_class.return_value = mock_dm_instance

        state = IISAState()

        with patch.dict(os.environ, {"IISA_GCP_PROJECT": "test-project"}):
            settings = Settings()

        # Act
        result = state.initialize(settings)

        # Assert
        assert result is True
        assert state._initialized is True
        assert state.settings is settings
        assert state.data_manager is mock_dm_instance
        mock_bq_class.assert_called_once_with(
            project="test-project",
            location="US",
        )
        mock_dm_class.assert_called_once_with(mock_bq_instance)

    @patch("iisa.iisa_http_endpoints.BigQueryProvider")
    def test_initialize_failure(self, mock_bq_class):
        """Mock BigQueryProvider to raise, verify returns False and logs warning."""
        # Arrange
        from iisa.iisa_http_endpoints import IISAState, Settings

        mock_bq_class.side_effect = Exception("Connection failed")

        state = IISAState()

        with patch.dict(os.environ, {"IISA_GCP_PROJECT": "test-project"}):
            settings = Settings()

        # Act
        result = state.initialize(settings)

        # Assert
        assert result is False
        assert state._initialized is False

    @patch("iisa.iisa_http_endpoints.DataManager")
    @patch("iisa.iisa_http_endpoints.BigQueryProvider")
    def test_refresh_data_success(self, mock_bq_class, mock_dm_class):
        """Mock load_scores()=True, verify _history populated."""
        # Arrange
        from iisa.iisa_http_endpoints import IISAState, Settings

        mock_history_df = pd.DataFrame({"indexer": ["0xABC", "0xXYZ"]})

        mock_dm_instance = MagicMock()
        mock_dm_instance.load_scores.return_value = True
        mock_dm_instance.get_data.return_value = mock_history_df
        mock_dm_class.return_value = mock_dm_instance

        state = IISAState()

        with patch.dict(os.environ, {"IISA_GCP_PROJECT": "test-project"}):
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
    @patch("iisa.iisa_http_endpoints.BigQueryProvider")
    def test_refresh_data_failure(self, mock_bq_class, mock_dm_class):
        """Mock load_scores()=False, verify returns False."""
        # Arrange
        from iisa.iisa_http_endpoints import IISAState, Settings

        mock_dm_instance = MagicMock()
        mock_dm_instance.load_scores.return_value = False
        mock_dm_class.return_value = mock_dm_instance

        state = IISAState()

        with patch.dict(os.environ, {"IISA_GCP_PROJECT": "test-project"}):
            settings = Settings()

        state.initialize(settings)

        # Act
        result = state.refresh_data()

        # Assert
        assert result is False

    @patch("iisa.iisa_http_endpoints.DataManager")
    @patch("iisa.iisa_http_endpoints.BigQueryProvider")
    def test_refresh_data_exception(self, mock_bq_class, mock_dm_class):
        """Mock load_scores() to raise exception, verify returns False."""
        # Arrange
        from iisa.iisa_http_endpoints import IISAState, Settings

        mock_dm_instance = MagicMock()
        mock_dm_instance.load_scores.side_effect = Exception("BigQuery connection failed")
        mock_dm_class.return_value = mock_dm_instance

        state = IISAState()

        with patch.dict(os.environ, {"IISA_GCP_PROJECT": "test-project"}):
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
def reset_state():
    """Reset global state before each test."""
    # Clear settings cache before test
    from iisa import iisa_http_endpoints

    iisa_http_endpoints.get_settings.cache_clear()

    # Reset state
    iisa_http_endpoints._state = iisa_http_endpoints.IISAState()

    yield

    # Clean up after test
    iisa_http_endpoints._state = iisa_http_endpoints.IISAState()
    iisa_http_endpoints.get_settings.cache_clear()


@pytest.fixture
def mock_history_df():
    """Create a mock DataFrame simulating loaded history data."""
    return pd.DataFrame({
        "indexer": ["0xABC", "0xXYZ", "0x123"],
        "url": ["https://a.com/", "https://b.com/", "https://c.com/"],
        "norm_lat_lin_reg_coefficient": [0.8, 0.9, 0.6],
        "norm_uptime_score": [0.9, 0.7, 0.95],
        "norm_success_rate": [0.85, 0.6, 0.9],
        "norm_stake_to_fees_iqr_deviation": [0.5, 0.8, 0.65],
        "existing_dips_agreements": [2, 1, 0],
    })


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


class TestRefreshEndpoint:
    """Tests for POST /refresh endpoint."""

    @patch("iisa.iisa_http_endpoints.DataManager")
    @patch("iisa.iisa_http_endpoints.BigQueryProvider")
    def test_refresh_success(self, mock_bq_class, mock_dm_class, mock_history_df):
        """Mock refresh_data()=True, verify 200 with row count."""
        # Arrange
        from iisa import iisa_http_endpoints
        from iisa.iisa_http_endpoints import Settings, app

        mock_dm_instance = MagicMock()
        mock_dm_instance.load_scores.return_value = True
        mock_dm_instance.get_data.return_value = mock_history_df
        mock_dm_class.return_value = mock_dm_instance

        with patch.dict(os.environ, {"IISA_GCP_PROJECT": "test-project"}):
            settings = Settings()
            iisa_http_endpoints._state.initialize(settings)

        client = TestClient(app, raise_server_exceptions=False)

        # Act
        response = client.post("/refresh")

        # Assert
        assert response.status_code == 200
        data = response.json()
        assert data["status"] == "success"
        assert data["rows"] == 3

    def test_refresh_not_initialized(self):
        """_initialized=False, verify 503 Service Unavailable."""
        # Arrange
        from iisa.iisa_http_endpoints import app

        client = TestClient(app, raise_server_exceptions=False)

        # Act
        response = client.post("/refresh")

        # Assert
        assert response.status_code == 503
        assert "not initialized" in response.json()["detail"]

    @patch("iisa.iisa_http_endpoints.DataManager")
    @patch("iisa.iisa_http_endpoints.BigQueryProvider")
    def test_refresh_failure(self, mock_bq_class, mock_dm_class):
        """Mock refresh_data()=False, verify 500 Internal Server Error."""
        # Arrange
        from iisa import iisa_http_endpoints
        from iisa.iisa_http_endpoints import Settings, app

        mock_dm_instance = MagicMock()
        mock_dm_instance.load_scores.return_value = False
        mock_dm_class.return_value = mock_dm_instance

        with patch.dict(os.environ, {"IISA_GCP_PROJECT": "test-project"}):
            settings = Settings()
            iisa_http_endpoints._state.initialize(settings)

        client = TestClient(app, raise_server_exceptions=False)

        # Act
        response = client.post("/refresh")

        # Assert
        assert response.status_code == 500
        assert "Failed to refresh" in response.json()["detail"]


class TestSelectOneEndpoint:
    """Tests for POST /select-one endpoint."""

    @patch("iisa.iisa_http_endpoints.DataProcessor")
    def test_select_one_intelligent_path(self, mock_processor_class, mock_history_df):
        """With data loaded, mock DataProcessor, verify selection."""
        # Arrange
        from iisa import iisa_http_endpoints
        from iisa.iisa_http_endpoints import app

        mock_processor = MagicMock()
        mock_processor.get_indexer_selections.return_value = (
            {"Qm123": ["0xABC"]},  # added
            {},  # cancelled
        )
        mock_processor_class.return_value = mock_processor

        iisa_http_endpoints._state._history = mock_history_df
        iisa_http_endpoints._state._initialized = True

        client = TestClient(app, raise_server_exceptions=False)

        # Act
        response = client.post(
            "/select-one",
            json={
                "deployment_id": "Qm123",
                "candidates": [
                    {"id": "0xABC", "url": "https://a.com/"},
                    {"id": "0xXYZ", "url": "https://b.com/"},
                ],
            },
        )

        # Assert
        assert response.status_code == 200
        data = response.json()
        assert data["indexer_id"] == "0xABC"

    def test_select_one_no_data_returns_503(self):
        """Without data loaded, verify 503 returned."""
        # Arrange
        from iisa import iisa_http_endpoints
        from iisa.iisa_http_endpoints import app

        iisa_http_endpoints._state._history = None
        iisa_http_endpoints._state._initialized = False

        client = TestClient(app, raise_server_exceptions=False)

        # Act
        response = client.post(
            "/select-one",
            json={
                "deployment_id": "Qm123",
                "candidates": [
                    {"id": "0xABC", "url": "https://a.com/"},
                    {"id": "0xXYZ", "url": "https://b.com/"},
                ],
            },
        )

        # Assert
        assert response.status_code == 503
        assert "IISA data not loaded" in response.json()["detail"]

    @patch("iisa.iisa_http_endpoints.DataProcessor")
    def test_select_one_empty_result(self, mock_processor_class, mock_history_df):
        """DataProcessor returns no selection, verify indexer_id=None."""
        # Arrange
        from iisa import iisa_http_endpoints
        from iisa.iisa_http_endpoints import app

        mock_processor = MagicMock()
        mock_processor.get_indexer_selections.return_value = (
            {},  # no added indexers
            {},  # no cancelled
        )
        mock_processor_class.return_value = mock_processor

        iisa_http_endpoints._state._history = mock_history_df
        iisa_http_endpoints._state._initialized = True

        client = TestClient(app, raise_server_exceptions=False)

        # Act
        response = client.post(
            "/select-one",
            json={
                "deployment_id": "Qm123",
            },
        )

        # Assert
        assert response.status_code == 200
        data = response.json()
        assert data["indexer_id"] is None

    @patch("iisa.iisa_http_endpoints.DataProcessor")
    def test_select_one_processor_exception_returns_500(
        self, mock_processor_class, mock_history_df
    ):
        """DataProcessor raises, verify 500 returned."""
        # Arrange
        from iisa import iisa_http_endpoints
        from iisa.iisa_http_endpoints import app

        mock_processor_class.side_effect = Exception("Processing failed")

        iisa_http_endpoints._state._history = mock_history_df
        iisa_http_endpoints._state._initialized = True

        client = TestClient(app, raise_server_exceptions=False)

        # Act
        response = client.post(
            "/select-one",
            json={
                "deployment_id": "Qm123",
            },
        )

        # Assert
        assert response.status_code == 500
        assert "Selection failed: Processing failed" in response.json()["detail"]


class TestSelectManyEndpoint:
    """Tests for POST /select-many endpoint."""

    @patch("iisa.iisa_http_endpoints.DataProcessor")
    def test_select_many_intelligent_path(self, mock_processor_class, mock_history_df):
        """With data loaded, mock DataProcessor, verify selections."""
        # Arrange
        from iisa import iisa_http_endpoints
        from iisa.iisa_http_endpoints import app

        mock_processor = MagicMock()
        mock_processor.get_indexer_selections.return_value = (
            {"Qm123": ["0xABC", "0xXYZ"]},  # added
            {},  # cancelled
        )
        mock_processor_class.return_value = mock_processor

        iisa_http_endpoints._state._history = mock_history_df
        iisa_http_endpoints._state._initialized = True

        client = TestClient(app, raise_server_exceptions=False)

        # Act
        response = client.post(
            "/select-many",
            json={
                "deployment_id": "Qm123",
                "candidates": [
                    {"id": "0xABC", "url": "https://a.com/"},
                    {"id": "0xXYZ", "url": "https://b.com/"},
                    {"id": "0x123", "url": "https://c.com/"},
                ],
                "num_candidates": 2,
            },
        )

        # Assert
        assert response.status_code == 200
        data = response.json()
        assert data["indexer_ids"] == ["0xABC", "0xXYZ"]

    def test_select_many_no_data_returns_503(self):
        """Without data loaded, verify 503 returned."""
        # Arrange
        from iisa import iisa_http_endpoints
        from iisa.iisa_http_endpoints import app

        iisa_http_endpoints._state._history = None
        iisa_http_endpoints._state._initialized = False

        client = TestClient(app, raise_server_exceptions=False)

        # Act
        response = client.post(
            "/select-many",
            json={
                "deployment_id": "Qm123",
                "num_candidates": 2,
            },
        )

        # Assert
        assert response.status_code == 503
        assert "IISA data not loaded" in response.json()["detail"]

    def test_select_many_missing_num_candidates(self, mock_history_df):
        """Verify 400 Bad Request when num_candidates missing."""
        # Arrange
        from iisa import iisa_http_endpoints
        from iisa.iisa_http_endpoints import app

        iisa_http_endpoints._state._history = mock_history_df
        iisa_http_endpoints._state._initialized = True

        client = TestClient(app, raise_server_exceptions=False)

        # Act
        response = client.post(
            "/select-many",
            json={
                "deployment_id": "Qm123",
            },
        )

        # Assert
        assert response.status_code == 400
        assert "num_candidates is required" in response.json()["detail"]

    @patch("iisa.iisa_http_endpoints.DataProcessor")
    def test_select_many_zero_num_candidates(self, mock_processor_class, mock_history_df):
        """Verify empty list returned when num_candidates is 0."""
        # Arrange
        from iisa import iisa_http_endpoints
        from iisa.iisa_http_endpoints import app

        iisa_http_endpoints._state._history = mock_history_df
        iisa_http_endpoints._state._initialized = True

        client = TestClient(app, raise_server_exceptions=False)

        # Act
        response = client.post(
            "/select-many",
            json={
                "deployment_id": "Qm123",
                "num_candidates": 0,
            },
        )

        # Assert
        assert response.status_code == 200
        data = response.json()
        assert data["indexer_ids"] == []
        mock_processor_class.assert_not_called()

    @patch("iisa.iisa_http_endpoints.DataProcessor")
    def test_select_many_processor_exception_returns_500(
        self, mock_processor_class, mock_history_df
    ):
        """DataProcessor raises, verify 500 returned."""
        # Arrange
        from iisa import iisa_http_endpoints
        from iisa.iisa_http_endpoints import app

        mock_processor_class.side_effect = Exception("Processing failed")

        iisa_http_endpoints._state._history = mock_history_df
        iisa_http_endpoints._state._initialized = True

        client = TestClient(app, raise_server_exceptions=False)

        # Act
        response = client.post(
            "/select-many",
            json={
                "deployment_id": "Qm123",
                "num_candidates": 2,
            },
        )

        # Assert
        assert response.status_code == 500
        assert "Selection failed: Processing failed" in response.json()["detail"]


class TestSelectWithProcessor:
    """Tests for _select_with_processor helper."""

    @patch("iisa.iisa_http_endpoints.DataProcessor")
    def test_select_with_processor_success(self, mock_processor_class, mock_history_df):
        """Mock DataProcessor.get_indexer_selections(), verify result."""
        # Arrange
        from iisa import iisa_http_endpoints
        from iisa.iisa_http_endpoints import SelectionRequest, _select_with_processor

        mock_processor = MagicMock()
        mock_processor.get_indexer_selections.return_value = (
            {"Qm123": ["0xABC", "0xXYZ", "0x123"]},
            {},
        )
        mock_processor_class.return_value = mock_processor

        iisa_http_endpoints._state._history = mock_history_df

        request = SelectionRequest(
            deployment_id="Qm123",
            candidates=[],
            existing_indexers=["0xEXIST"],
        )

        # Act
        result = _select_with_processor(request, num_to_select=2)

        # Assert
        assert result == ["0xABC", "0xXYZ"]
        mock_processor_class.assert_called_once()

    def test_select_with_processor_no_history(self):
        """_state.history=None, verify empty list."""
        # Arrange
        from iisa import iisa_http_endpoints
        from iisa.iisa_http_endpoints import SelectionRequest, _select_with_processor

        iisa_http_endpoints._state._history = None

        request = SelectionRequest(deployment_id="Qm123")

        # Act
        result = _select_with_processor(request, num_to_select=2)

        # Assert
        assert result == []

    @patch("iisa.iisa_http_endpoints.DataProcessor")
    def test_select_with_processor_maps_blocklist(
        self, mock_processor_class, mock_history_df
    ):
        """Verify blocklist mapped to indexer_denylist param."""
        # Arrange
        from iisa import iisa_http_endpoints
        from iisa.iisa_http_endpoints import SelectionRequest, _select_with_processor

        mock_processor = MagicMock()
        mock_processor.get_indexer_selections.return_value = ({"Qm123": []}, {})
        mock_processor_class.return_value = mock_processor

        iisa_http_endpoints._state._history = mock_history_df

        request = SelectionRequest(
            deployment_id="Qm123",
            blocklist=["0xBAD1", "0xBAD2"],
        )

        # Act
        _select_with_processor(request, num_to_select=1)

        # Assert
        call_kwargs = mock_processor_class.call_args[1]
        assert call_kwargs["indexer_denylist"] == ["0xBAD1", "0xBAD2"]

    @patch("iisa.iisa_http_endpoints.DataProcessor")
    def test_select_with_processor_builds_existing_agreements(
        self, mock_processor_class, mock_history_df
    ):
        """Verify existing_indexers mapped to existing_agreements dict."""
        # Arrange
        from iisa import iisa_http_endpoints
        from iisa.iisa_http_endpoints import SelectionRequest, _select_with_processor

        mock_processor = MagicMock()
        mock_processor.get_indexer_selections.return_value = ({"Qm123": []}, {})
        mock_processor_class.return_value = mock_processor

        iisa_http_endpoints._state._history = mock_history_df

        request = SelectionRequest(
            deployment_id="Qm123",
            existing_indexers=["0xEXIST1", "0xEXIST2"],
        )

        # Act
        _select_with_processor(request, num_to_select=1)

        # Assert
        call_kwargs = mock_processor_class.call_args[1]
        assert call_kwargs["existing_agreements"] == {"Qm123": ["0xEXIST1", "0xEXIST2"]}

    @patch("iisa.iisa_http_endpoints.DataProcessor")
    def test_select_with_processor_passes_pending_agreements(
        self, mock_processor_class, mock_history_df
    ):
        """Verify pending_agreements passed through."""
        # Arrange
        from iisa import iisa_http_endpoints
        from iisa.iisa_http_endpoints import SelectionRequest, _select_with_processor

        mock_processor = MagicMock()
        mock_processor.get_indexer_selections.return_value = ({"Qm123": []}, {})
        mock_processor_class.return_value = mock_processor

        iisa_http_endpoints._state._history = mock_history_df

        pending = {"Qm123": ["0xPEND1"], "Qm456": ["0xPEND2"]}
        request = SelectionRequest(
            deployment_id="Qm123",
            pending_agreements=pending,
        )

        # Act
        _select_with_processor(request, num_to_select=1)

        # Assert
        call_kwargs = mock_processor_class.call_args[1]
        assert call_kwargs["pending_agreements"] == pending

    @patch("iisa.iisa_http_endpoints.DataProcessor")
    def test_select_with_processor_passes_declined_indexers(
        self, mock_processor_class, mock_history_df
    ):
        """Verify declined_indexers passed through."""
        # Arrange
        from iisa import iisa_http_endpoints
        from iisa.iisa_http_endpoints import SelectionRequest, _select_with_processor

        mock_processor = MagicMock()
        mock_processor.get_indexer_selections.return_value = ({"Qm123": []}, {})
        mock_processor_class.return_value = mock_processor

        iisa_http_endpoints._state._history = mock_history_df

        declined = {"Qm123": ["0xDEC1", "0xDEC2"]}
        request = SelectionRequest(
            deployment_id="Qm123",
            declined_indexers=declined,
        )

        # Act
        _select_with_processor(request, num_to_select=1)

        # Assert
        call_kwargs = mock_processor_class.call_args[1]
        assert call_kwargs["declined_indexers"] == declined
