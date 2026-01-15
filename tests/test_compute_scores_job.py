"""
Test suite for the score computation CronJob.

Tests the new functions introduced for the CronJob, particularly:
- Normalization functions
- Schema transformation
- GeoIP utilities
- Idempotency check
"""

from datetime import datetime
from unittest.mock import MagicMock, patch

import numpy as np
import pandas as pd
import pytest

# Import from the jobs package - we need to add it to the path
import sys
from pathlib import Path

jobs_path = Path(__file__).parent.parent / "jobs" / "compute_scores"
sys.path.insert(0, str(jobs_path))

from processing import (
    is_private_ip,
    normalize_to_0_1,
    normalize_to_0_1_inverted,
    normalize_iqr_to_0_1,
    transform_to_scores_schema,
    haversine_vectorized,
    filter_successful_queries,
    iterative_filter,
    calculate_indexer_success_rate,
    calculate_indexer_stake_to_fees,
)
from bq import BigQueryClient


class TestNormalizeToZeroOne:
    """Tests for the normalize_to_0_1 function."""

    def test_normalize_basic(self):
        # Arrange
        series = pd.Series([0, 50, 100])

        # Act
        result = normalize_to_0_1(series)

        # Assert
        expected = pd.Series([0.0, 0.5, 1.0])
        pd.testing.assert_series_equal(result, expected)

    def test_normalize_negative_values(self):
        # Arrange
        series = pd.Series([-100, 0, 100])

        # Act
        result = normalize_to_0_1(series)

        # Assert
        expected = pd.Series([0.0, 0.5, 1.0])
        pd.testing.assert_series_equal(result, expected)

    def test_normalize_constant_values(self):
        # Arrange
        series = pd.Series([5, 5, 5])

        # Act
        result = normalize_to_0_1(series)

        # Assert - all values should be 0.5 when constant
        expected = pd.Series([0.5, 0.5, 0.5])
        pd.testing.assert_series_equal(result, expected)

    def test_normalize_empty_series(self):
        # Arrange
        series = pd.Series([], dtype=float)

        # Act
        result = normalize_to_0_1(series)

        # Assert
        assert result.empty

    def test_normalize_single_value(self):
        # Arrange
        series = pd.Series([42])

        # Act
        result = normalize_to_0_1(series)

        # Assert - single value normalizes to 0.5
        expected = pd.Series([0.5])
        pd.testing.assert_series_equal(result, expected)

    def test_normalize_with_nan(self):
        # Arrange
        series = pd.Series([0, np.nan, 100])

        # Act
        result = normalize_to_0_1(series)

        # Assert - NaN should be preserved
        assert result.iloc[0] == 0.0
        assert pd.isna(result.iloc[1])
        assert result.iloc[2] == 1.0


class TestNormalizeToZeroOneInverted:
    """Tests for the normalize_to_0_1_inverted function."""

    def test_normalize_inverted_basic(self):
        # Arrange
        series = pd.Series([0, 50, 100])

        # Act
        result = normalize_to_0_1_inverted(series)

        # Assert - inverted so 0 -> 1.0, 100 -> 0.0
        expected = pd.Series([1.0, 0.5, 0.0])
        pd.testing.assert_series_equal(result, expected)

    def test_normalize_inverted_latency_use_case(self):
        # Arrange - lower latency should result in higher score
        latency_ms = pd.Series([100, 200, 300])

        # Act
        result = normalize_to_0_1_inverted(latency_ms)

        # Assert - lowest latency (100) gets highest score (1.0)
        assert result.iloc[0] == 1.0
        assert result.iloc[2] == 0.0

    def test_normalize_inverted_empty(self):
        # Arrange
        series = pd.Series([], dtype=float)

        # Act
        result = normalize_to_0_1_inverted(series)

        # Assert
        assert result.empty


class TestNormalizeIqrToZeroOne:
    """Tests for the normalize_iqr_to_0_1 function."""

    def test_normalize_iqr_basic(self):
        # Arrange - IQR deviations can be negative
        series = pd.Series([-2, -1, 0, 1, 2])

        # Act
        result = normalize_iqr_to_0_1(series)

        # Assert
        assert result.min() == 0.0
        assert result.max() == 1.0

    def test_normalize_iqr_empty(self):
        # Arrange
        series = pd.Series([], dtype=float)

        # Act
        result = normalize_iqr_to_0_1(series)

        # Assert
        assert result.empty


class TestTransformToScoresSchema:
    """Tests for the transform_to_scores_schema function."""

    @pytest.fixture
    def sample_merged_df(self):
        return pd.DataFrame({
            "indexer": ["0xABC", "0xDEF", "0x123"],
            "url": ["https://a.com", "https://b.com", "https://c.com"],
            "Latency Coefficient": [0.5, 1.0, 1.5],
            "Standard Error": [0.1, 0.2, 0.3],
            "Latency Coefficient + Error Confidence Interval": [0.65, 1.3, 1.95],
            "Robust Normalized Latency Coefficient + Error Confidence Interval": [-0.5, 0, 0.5],
            "% up_x": [99.5, 95.0, 90.0],
            "observed_duration_restricted": [1000, 2000, 3000],
            "uptime_duration_restricted": [995, 1900, 2700],
            "average_status": [0.98, 0.95, 0.90],
            "stake_to_fees": [100.0, 200.0, 300.0],
            "stake_to_fees_iqr_deviation": [-1.0, 0.0, 1.0],
            "org": ["AWS", "GCP", "Hetzner"],
            "dst_lat": [40.0, 35.0, 50.0],
            "dst_lon": [-74.0, -120.0, 8.0],
        })

    def test_transform_basic(self, sample_merged_df):
        # Act
        result = transform_to_scores_schema(sample_merged_df, num_days=28)

        # Assert - check required columns exist
        required_columns = [
            "indexer", "url",
            "lat_lin_reg_coefficient", "lat_coefficient_std_error", "lat_coefficient_upper_bound",
            "lat_normalized_score",
            "uptime_score", "observed_duration_seconds", "uptime_duration_seconds",
            "success_rate",
            "stake_to_fees", "stake_to_fees_iqr_deviation",
            "norm_uptime_score", "norm_success_rate", "norm_stake_to_fees",
            "org", "dst_lat", "dst_lon",
            "computed_at", "query_count", "num_days",
        ]
        for col in required_columns:
            assert col in result.columns, f"Missing column: {col}"

    def test_transform_uptime_conversion(self, sample_merged_df):
        # Act
        result = transform_to_scores_schema(sample_merged_df, num_days=28)

        # Assert - uptime should be converted from percentage to 0-1 scale
        assert result["uptime_score"].iloc[0] == pytest.approx(0.995)
        assert result["uptime_score"].iloc[1] == pytest.approx(0.95)

    def test_transform_num_days(self, sample_merged_df):
        # Act
        result = transform_to_scores_schema(sample_merged_df, num_days=14)

        # Assert
        assert all(result["num_days"] == 14)

    def test_transform_computed_at_is_recent(self, sample_merged_df):
        # Act
        result = transform_to_scores_schema(sample_merged_df, num_days=28)

        # Assert - computed_at should be recent
        now = datetime.utcnow()
        computed_at = result["computed_at"].iloc[0]
        delta = (now - computed_at).total_seconds()
        assert delta < 60  # Within 60 seconds

    def test_transform_normalized_scores_bounded(self, sample_merged_df):
        # Act
        result = transform_to_scores_schema(sample_merged_df, num_days=28)

        # Assert - all normalized scores should be between 0 and 1
        for col in ["norm_uptime_score", "norm_success_rate", "norm_stake_to_fees", "lat_normalized_score"]:
            values = result[col].dropna()
            assert all(0 <= v <= 1 for v in values), f"{col} has values outside [0, 1]"


class TestIsPrivateIp:
    """Tests for the is_private_ip function."""

    def test_localhost(self):
        assert is_private_ip("127.0.0.1") is True
        assert is_private_ip("127.255.255.255") is True

    def test_class_a_private(self):
        assert is_private_ip("10.0.0.1") is True
        assert is_private_ip("10.255.255.255") is True

    def test_class_b_private(self):
        assert is_private_ip("172.16.0.1") is True
        assert is_private_ip("172.31.255.255") is True
        assert is_private_ip("172.15.0.1") is False  # Just outside range
        assert is_private_ip("172.32.0.1") is False  # Just outside range

    def test_class_c_private(self):
        assert is_private_ip("192.168.0.1") is True
        assert is_private_ip("192.168.255.255") is True
        assert is_private_ip("192.167.0.1") is False  # Just outside range

    def test_public_ips(self):
        assert is_private_ip("8.8.8.8") is False  # Google DNS
        assert is_private_ip("1.1.1.1") is False  # Cloudflare DNS
        assert is_private_ip("142.250.80.46") is False  # Google


class TestHaversineVectorized:
    """Tests for the haversine_vectorized function."""

    def test_same_location(self):
        # Arrange
        lon1 = pd.Series([0.0])
        lat1 = pd.Series([0.0])
        lon2 = pd.Series([0.0])
        lat2 = pd.Series([0.0])

        # Act
        result = haversine_vectorized(lon1, lat1, lon2, lat2)

        # Assert
        assert result[0] == 0.0

    def test_known_distance(self):
        # Arrange - NYC to LA approx 2450 miles
        nyc_lon, nyc_lat = -74.006, 40.7128
        la_lon, la_lat = -118.2437, 34.0522

        # Act
        result = haversine_vectorized(
            pd.Series([nyc_lon]), pd.Series([nyc_lat]),
            pd.Series([la_lon]), pd.Series([la_lat])
        )

        # Assert - should be approximately 2450 miles
        assert 2400 < result[0] < 2500

    def test_vectorized_multiple_points(self):
        # Arrange
        lon1 = pd.Series([0.0, 0.0])
        lat1 = pd.Series([0.0, 0.0])
        lon2 = pd.Series([0.0, 30.0])
        lat2 = pd.Series([0.0, 0.0])

        # Act
        result = haversine_vectorized(lon1, lat1, lon2, lat2)

        # Assert
        assert len(result) == 2
        assert result[0] == 0.0
        assert result[1] > 0


class TestFilterSuccessfulQueries:
    """Tests for filter_successful_queries function."""

    def test_filter_200_ok(self):
        # Arrange
        df = pd.DataFrame({
            "status": ["200 OK", "404 Not Found", "200 OK", "500 Error"],
            "data": ["a", "b", "c", "d"],
        })

        # Act
        result = filter_successful_queries(df)

        # Assert
        assert len(result) == 2
        assert list(result["data"]) == ["a", "c"]

    def test_filter_empty_df(self):
        # Arrange
        df = pd.DataFrame(columns=["status", "data"])

        # Act
        result = filter_successful_queries(df)

        # Assert
        assert result.empty


class TestCalculateIndexerSuccessRate:
    """Tests for calculate_indexer_success_rate function."""

    def test_success_rate_basic(self):
        # Arrange
        df = pd.DataFrame({
            "indexer": ["A", "A", "A", "B", "B"],
            "status": ["200 OK", "200 OK", "Error", "200 OK", "200 OK"],
        })

        # Act
        result = calculate_indexer_success_rate(df)

        # Assert
        assert len(result) == 2
        a_rate = result[result["indexer"] == "A"]["average_status"].iloc[0]
        b_rate = result[result["indexer"] == "B"]["average_status"].iloc[0]
        assert a_rate == pytest.approx(2/3)
        assert b_rate == pytest.approx(1.0)

    def test_success_rate_includes_missing_block(self):
        # Arrange - Unavailable(MissingBlock) counts as success
        df = pd.DataFrame({
            "indexer": ["A", "A"],
            "status": ["200 OK", "Unavailable(MissingBlock)"],
        })

        # Act
        result = calculate_indexer_success_rate(df)

        # Assert
        assert result["average_status"].iloc[0] == 1.0


class TestCalculateIndexerStakeToFees:
    """Tests for calculate_indexer_stake_to_fees function."""

    def test_stake_to_fees_basic(self):
        # Arrange
        df = pd.DataFrame({
            "indexer": ["A", "B", "C", "D", "E"],
            "stake_to_fees": [1.0, 2.0, 3.0, 4.0, 5.0],
        })
        df = df.set_index("indexer")

        # Act
        result = calculate_indexer_stake_to_fees(df)

        # Assert
        assert "stake_to_fees_iqr_deviation" in result.columns
        assert len(result) == 5

    def test_stake_to_fees_iqr_calculation(self):
        # Arrange
        df = pd.DataFrame({
            "indexer": ["A", "B", "C", "D", "E"],
            "stake_to_fees": [1.0, 2.0, 3.0, 4.0, 5.0],
        })
        df = df.set_index("indexer")

        # Act
        result = calculate_indexer_stake_to_fees(df)

        # Assert - median is 3.0, Q1=2.0, Q3=4.0, IQR=2.0
        # Deviation for C (value=3.0) should be (3.0 - 3.0) / 2.0 = 0.0
        c_deviation = result[result["indexer"] == "C"]["stake_to_fees_iqr_deviation"].iloc[0]
        assert c_deviation == pytest.approx(0.0)


class TestIterativeFilter:
    """Tests for iterative_filter function."""

    @pytest.fixture
    def sample_df(self):
        return pd.DataFrame({
            "deployment_hash": ["A"] * 10 + ["B"] * 10,
            "indexer": ["X", "Y"] * 10,
            "query_id": list(range(20)),
        })

    def test_iterative_filter_no_change(self, sample_df):
        # Act
        result = iterative_filter(
            sample_df,
            min_deployment_indexers=1,
            min_deployments_per_indexer=1,
            min_queries_per_indexer=1,
            min_queries_per_deployment=1,
        )

        # Assert
        assert len(result) == len(sample_df)

    def test_iterative_filter_removes_sparse_data(self, sample_df):
        # Act - require 3 indexers per deployment (we only have 2)
        result = iterative_filter(
            sample_df,
            min_deployment_indexers=3,
            min_deployments_per_indexer=1,
            min_queries_per_indexer=1,
            min_queries_per_deployment=1,
        )

        # Assert
        assert len(result) == 0

    def test_iterative_filter_empty_df(self):
        # Arrange
        df = pd.DataFrame(columns=["deployment_hash", "indexer", "query_id"])

        # Act
        result = iterative_filter(df, 0, 0, 0, 0)

        # Assert
        assert len(result) == 0


class TestBigQueryClientIdempotency:
    """Tests for BigQuery client idempotency check."""

    def test_scores_exist_returns_true_when_data_exists(self):
        # Arrange
        with patch("bq.bpd") as mock_bpd:
            mock_bpd.options.bigquery.project = None
            mock_bpd.options.bigquery.location = None
            mock_bpd.options.display.progress_bar = None

            client = BigQueryClient(
                project="test-project",
                dataset="test-dataset",
                location="US",
            )

            # Mock the read_gbq to return a count > 0
            mock_df = pd.DataFrame({"cnt": [5]})
            mock_bpd.read_gbq.return_value.to_pandas.return_value = mock_df

            # Act
            result = client.scores_exist_for_today()

            # Assert
            assert result == True

    def test_scores_exist_returns_false_when_no_data(self):
        # Arrange
        with patch("bq.bpd") as mock_bpd:
            mock_bpd.options.bigquery.project = None
            mock_bpd.options.bigquery.location = None
            mock_bpd.options.display.progress_bar = None

            client = BigQueryClient(
                project="test-project",
                dataset="test-dataset",
                location="US",
            )

            # Mock the read_gbq to return count = 0
            mock_df = pd.DataFrame({"cnt": [0]})
            mock_bpd.read_gbq.return_value.to_pandas.return_value = mock_df

            # Act
            result = client.scores_exist_for_today()

            # Assert
            assert result == False

    def test_scores_exist_returns_false_on_error(self):
        # Arrange
        with patch("bq.bpd") as mock_bpd:
            mock_bpd.options.bigquery.project = None
            mock_bpd.options.bigquery.location = None
            mock_bpd.options.display.progress_bar = None

            client = BigQueryClient(
                project="test-project",
                dataset="test-dataset",
                location="US",
            )

            # Mock read_gbq to raise an exception (table doesn't exist)
            mock_bpd.read_gbq.return_value.to_pandas.side_effect = Exception("Table not found")

            # Act
            result = client.scores_exist_for_today()

            # Assert - should return False on error (graceful handling)
            assert result is False
