"""Tests for the DataManager class."""

from datetime import datetime, timedelta, timezone
from unittest.mock import MagicMock

import numpy as np
import pandas as pd
import pytest

from iisa import DataManager


class TestDataManager:
    """Tests for DataManager."""

    @pytest.fixture
    def mock_scores_df(self):
        """Create a mock DataFrame simulating the indexer_scores table."""
        return pd.DataFrame({
            "indexer": ["0xABC", "0xXYZ", "0x123"],
            "url": ["https://a.com/", "https://b.com/", "https://c.com/"],
            "lat_lin_reg_coefficient": [0.5, 0.3, 0.7],
            "lat_coefficient_std_error": [0.05, 0.03, 0.07],
            "lat_coefficient_upper_bound": [0.55, 0.33, 0.77],
            "lat_normalized_score": [0.8, 0.9, 0.6],
            "uptime_score": [0.98, 0.95, 0.99],
            "success_rate": [0.95, 0.90, 0.97],
            "stake_to_fees": [1000, 2000, 1500],
            "stake_to_fees": [0.5, 1.0, 0.75],
            "norm_uptime_score": [0.9, 0.7, 0.95],
            "norm_success_rate": [0.85, 0.6, 0.9],
            "norm_stake_to_fees": [0.5, 0.8, 0.65],
            "org": ["hetzner", "amazon", "google"],
            "dst_lat": [50.0, 40.0, 35.0],
            "dst_lon": [10.0, -74.0, 139.0],
            "avg_sync_duration": [np.nan, np.nan, np.nan],
            "computed_at": [datetime(2024, 1, 15, 12, 0)] * 3,
            "query_count": [10000, 8000, 12000],
            "num_days": [28, 28, 28],
        })

    @pytest.fixture
    def mock_provider_with_scores(self, mock_scores_df):
        """Create a mock provider that returns scores."""
        mock_provider = MagicMock()
        mock_provider.fetch_indexer_scores.return_value = (
            mock_scores_df,
            datetime(2024, 1, 15, 12, 0),
        )
        return mock_provider

    @pytest.fixture
    def mock_provider_empty_scores(self):
        """Create a mock provider that returns empty scores."""
        mock_provider = MagicMock()
        mock_provider.fetch_indexer_scores.return_value = (pd.DataFrame(), None)
        return mock_provider

    def test_load_scores_success(self, mock_provider_with_scores):
        """Test that load_scores() successfully loads and transforms data."""
        data_manager = DataManager(mock_provider_with_scores)

        result = data_manager.load_scores()

        assert result is True
        data = data_manager.get_data()
        assert data is not None
        assert len(data) == 3

    def test_load_scores_column_transformation(self, mock_provider_with_scores):
        """Test that load_scores() correctly transforms column names."""
        data_manager = DataManager(mock_provider_with_scores)
        data_manager.load_scores()

        data = data_manager.get_data()

        assert "Latency Coefficient + Error Confidence Interval" in data.columns
        assert "average_status" in data.columns
        assert "% up_x" in data.columns
        assert "destination_loc" in data.columns
        assert "norm_lat_lin_reg_coefficient" in data.columns
        assert "norm_stake_to_fees" in data.columns

    def test_load_scores_uptime_conversion(self, mock_provider_with_scores):
        """Test that uptime is converted from 0-1 to percentage."""
        data_manager = DataManager(mock_provider_with_scores)
        data_manager.load_scores()

        data = data_manager.get_data()

        # Original was 0.98, 0.95, 0.99 -> should become 98, 95, 99
        assert data["% up_x"].iloc[0] == pytest.approx(98.0)
        assert data["% up_x"].iloc[1] == pytest.approx(95.0)
        assert data["% up_x"].iloc[2] == pytest.approx(99.0)

    def test_load_scores_destination_loc_creation(self, mock_provider_with_scores):
        """Test that destination_loc is created from lat/lon."""
        data_manager = DataManager(mock_provider_with_scores)
        data_manager.load_scores()

        data = data_manager.get_data()

        assert data["destination_loc"].iloc[0] == "50.0,10.0"
        assert data["destination_loc"].iloc[1] == "40.0,-74.0"
        assert data["destination_loc"].iloc[2] == "35.0,139.0"

    def test_load_scores_empty_table(self, mock_provider_empty_scores):
        """Test that load_scores() returns False when table is empty."""
        data_manager = DataManager(mock_provider_empty_scores)

        result = data_manager.load_scores()

        assert result is False
        assert data_manager.get_data() is None

    def test_load_scores_staleness_check(self, mock_provider_with_scores, caplog):
        """Test that staleness warnings are logged for old scores."""
        old_timestamp = datetime.now(timezone.utc) - timedelta(hours=100)
        mock_provider_with_scores.fetch_indexer_scores.return_value = (
            mock_provider_with_scores.fetch_indexer_scores.return_value[0],
            old_timestamp,
        )
        data_manager = DataManager(mock_provider_with_scores)

        import logging
        with caplog.at_level(logging.WARNING):
            data_manager.load_scores()

        assert any("stale" in record.message.lower() for record in caplog.records)

    def test_get_scores_age(self, mock_provider_with_scores):
        """Test that get_scores_age() returns correct age."""
        data_manager = DataManager(mock_provider_with_scores)

        assert data_manager.get_scores_age() is None

        data_manager.load_scores()

        age = data_manager.get_scores_age()
        assert age is not None
        assert isinstance(age, timedelta)

    def test_load_scores_preserves_precomputed_normalized_values(
        self, mock_provider_with_scores
    ):
        """Test that pre-computed normalized values are preserved."""
        data_manager = DataManager(mock_provider_with_scores)
        data_manager.load_scores()

        data = data_manager.get_data()

        assert "norm_uptime_score" in data.columns
        assert "norm_success_rate" in data.columns
        assert data["norm_uptime_score"].iloc[0] == pytest.approx(0.9)
        assert data["norm_success_rate"].iloc[0] == pytest.approx(0.85)
