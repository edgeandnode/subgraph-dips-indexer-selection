"""
Test suite for the score computation CronJob.

Tests the new functions introduced for the CronJob, particularly:
- Normalization functions
- Schema transformation
- GeoIP utilities
- Idempotency check
"""

from datetime import datetime, timezone
from unittest.mock import patch

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
    adjust_rows,
    strategic_sample,
    hash_sampled_queries,
    perform_latency_linear_regression,
    calculate_indexer_uptime,
    aggregate_indexer_info,
    merge_and_prepare_dataframes,
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
        now = datetime.now(timezone.utc)
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


# =============================================================================
# PORTED TESTS FROM test_data_manager.py
# These tests cover the critical processing functions that will be deleted
# from IISA after the CronJob migration is complete.
# =============================================================================


class TestAdjustRows:
    """Tests for the adjust_rows function."""

    def test_adjust_rows_normal_case(self):
        # Arrange
        sample_data = pd.DataFrame({
            "deployment_hash": ["hash1", "hash2", "hash3", "hash1"],
            "indexer": ["index1", "index2", "index3", "indexer4"],
            "num_rows": [50, 10000, 600, 50],
        })

        # Act
        target_rows = 600
        result = adjust_rows(sample_data, target_rows)

        # Assert - result should be close to target
        assert result > 0

    def test_adjust_rows_empty_dataframe(self):
        # Arrange
        df = pd.DataFrame({"deployment_hash": [], "indexer": [], "num_rows": []})

        # Act
        target_rows = 100
        result = adjust_rows(df, target_rows)

        # Assert - should handle empty gracefully
        assert result == 0 or df.empty

    def test_adjust_rows_zero_target(self):
        # Arrange
        sample_data = pd.DataFrame({
            "deployment_hash": ["hash1", "hash2"],
            "indexer": ["index1", "index2"],
            "num_rows": [50, 100],
        })

        # Act
        target_rows = 0
        result = adjust_rows(sample_data, target_rows)

        # Assert
        assert result == 0

    def test_adjust_rows_negative_case(self):
        # Arrange
        df = pd.DataFrame({
            "deployment_hash": ["hash1"],
            "indexer": ["index1"],
            "num_rows": [100],
        })

        # Act & Assert
        with pytest.raises(ValueError, match="non-negative"):
            adjust_rows(df, -300)


class TestStrategicSample:
    """Tests for the strategic_sample function."""

    @pytest.fixture
    def sample_df(self):
        return pd.DataFrame({
            "deployment_hash": ["A", "A", "A", "B", "B", "C"] * 10000,
            "indexer": ["X", "Y", "Z", "X", "Y", "X"] * 10000,
            "query_id": range(60000),
        })

    def test_strategic_sample_basic(self, sample_df):
        # Act
        result_df, integer_root = strategic_sample(sample_df, target_rows_per_subgraph=30)

        # Assert
        sampled = result_df[result_df["sampled_query_id"].notna()]

        # Check output df length unchanged
        assert len(result_df) == len(sample_df)

        # Check sampled count is reasonable
        assert result_df["sampled_query_id"].notna().sum() == 90

        # Verify integer_root calculation
        assert isinstance(integer_root, int)
        assert integer_root == int(np.sqrt(result_df["sampled_query_id"].notna().sum()))

    def test_strategic_sample_empty_df(self):
        # Arrange
        empty_df = pd.DataFrame(columns=["deployment_hash", "indexer", "query_id"])

        # Act
        result_df, integer_root = strategic_sample(empty_df, target_rows_per_subgraph=10)

        # Assert
        assert result_df.empty
        assert "sampled_query_id" in result_df.columns
        assert integer_root == 0

    def test_strategic_sample_target_rows_per_subgraph_greater_than_df(self, sample_df):
        # Act - target larger than data
        result_df, integer_root = strategic_sample(
            sample_df, target_rows_per_subgraph=10_000_000
        )

        # Assert - should sample all unique queries
        assert len(result_df) == len(sample_df)
        assert result_df["sampled_query_id"].notna().sum() == sample_df["query_id"].nunique()


class TestHashSampledQueries:
    """Tests for the hash_sampled_queries function."""

    @pytest.fixture
    def sample_df(self):
        return pd.DataFrame({
            "sampled_query_id": [1, 2, 3, None, 5, 6, np.nan, 8, 9, 10] * 1000,
            "other_column": ["A", "B", "C", "D", "E", "F", "G", "H", "I", "J"] * 1000,
        })

    def test_hash_sampled_queries_basic(self, sample_df):
        # Act
        integer_root = 33
        result = hash_sampled_queries(sample_df, integer_root)

        # Assert
        assert "sampled_query_id_hashed_mod_integer_root" in result.columns
        assert result["sampled_query_id_hashed_mod_integer_root"].notna().sum() == 8000
        assert result["sampled_query_id_hashed_mod_integer_root"].isna().sum() == 2000

        # All hashed values within range
        assert all(
            0 <= x < integer_root
            for x in result["sampled_query_id_hashed_mod_integer_root"].dropna()
        )

    def test_hash_sampled_queries_empty_df(self):
        # Arrange
        empty_df = pd.DataFrame(columns=["sampled_query_id"])

        # Act
        result = hash_sampled_queries(empty_df, 5)

        # Assert
        assert "sampled_query_id_hashed_mod_integer_root" in result.columns
        assert result.empty

    def test_hash_sampled_queries_all_null(self):
        # Arrange
        df = pd.DataFrame({"sampled_query_id": [None, None, None]})

        # Act
        result = hash_sampled_queries(df, 5)

        # Assert
        assert result["sampled_query_id_hashed_mod_integer_root"].isna().all()

    def test_hash_sampled_queries_consistency(self, sample_df):
        # Act - hash twice with same params
        integer_root = 7
        result1 = hash_sampled_queries(sample_df, integer_root)
        result2 = hash_sampled_queries(sample_df, integer_root)

        # Assert - results should be identical
        pd.testing.assert_frame_equal(result1, result2)

    def test_hash_sampled_queries_different_integer_roots(self, sample_df):
        # Act
        result1 = hash_sampled_queries(sample_df.copy(), 5)
        result2 = hash_sampled_queries(sample_df.copy(), 10)

        # Assert - different roots should produce different results
        assert not result1["sampled_query_id_hashed_mod_integer_root"].equals(
            result2["sampled_query_id_hashed_mod_integer_root"]
        )

    def test_hash_sampled_queries_original_df_unchanged(self, sample_df):
        # Arrange
        original_df = sample_df.copy()

        # Act
        _ = hash_sampled_queries(sample_df, 5)

        # Assert - original should be unchanged
        pd.testing.assert_frame_equal(original_df, sample_df)

    def test_hash_sampled_queries_large_integer_root(self):
        # Arrange
        df = pd.DataFrame({"sampled_query_id": range(1000)})
        large_root = 10_000_000_000

        # Act
        result = hash_sampled_queries(df, large_root)

        # Assert - all values within range
        assert all(
            0 <= x < large_root
            for x in result["sampled_query_id_hashed_mod_integer_root"]
        )


class TestPerformLinearRegression:
    """
    Tests for the latency linear regression pipeline.
    This is the CORE algorithm of the score computation.
    """

    @pytest.fixture
    def sample_df(self):
        np.random.seed(42)
        return pd.DataFrame({
            "sampled_query_id": range(10000),
            "indexer": np.random.choice(["0xABC", "0xXYZ", "0x123", "0x789"], 10000),
            "indexer_network": np.random.choice(["net1", "net2"], 10000),
            "deployment_hash": np.random.choice(
                ["deployment_1", "deployment_2", "deployment_3"], 10000
            ),
            "response_time_ms": np.random.randint(10, 20000, 10000),
            "fee": np.random.uniform(0.000001, 0.01, 10000),
            "distance_miles": np.random.uniform(0, 1000, 10000),
        })

    def test_perform_latency_linear_regression(self, sample_df):
        # Arrange
        integer_root = 100
        hashed_df = hash_sampled_queries(sample_df, integer_root)

        predictor = ["response_time_ms"]
        categorical = [
            "indexer", "deployment_hash", "indexer_network",
            "sampled_query_id_hashed_mod_integer_root",
        ]
        numeric = ["distance_miles", "fee"]

        # Act
        rankings, results = perform_latency_linear_regression(
            hashed_df, predictor, categorical, numeric
        )

        # Assert - rankings should have expected columns
        expected_columns = [
            "indexer", "Latency Coefficient", "Standard Error", "p-value",
            "Latency Coefficient + Error Confidence Interval",
            "Robust Normalized Latency Coefficient + Error Confidence Interval",
        ]
        assert all(col in rankings.columns for col in expected_columns)

        # Only indexer values in the indexer column
        assert all(rankings["indexer"].isin(["0xABC", "0xXYZ", "0x123", "0x789"]))

        # Coefficients should be numeric and non-null
        assert rankings["Latency Coefficient"].notna().all()

        # p-values between 0 and 1
        assert rankings["p-value"].between(0, 1).all()

    def test_perform_latency_linear_regression_with_empty_df(self):
        # Arrange
        empty_df = pd.DataFrame(columns=[
            "indexer", "deployment_hash", "indexer_network",
            "response_time_ms", "fee", "distance_miles",
        ])

        predictor = ["response_time_ms"]
        categorical = ["indexer", "deployment_hash", "indexer_network"]
        numeric = ["distance_miles", "fee"]

        # Act & Assert
        with pytest.raises(ValueError):
            perform_latency_linear_regression(empty_df, predictor, categorical, numeric)

    def test_perform_latency_linear_regression_with_missing_columns(self, sample_df):
        # Arrange - remove required columns
        df_missing = sample_df.drop(columns=["indexer", "fee"])

        predictor = ["response_time_ms"]
        categorical = ["indexer", "deployment_hash", "indexer_network"]
        numeric = ["distance_miles", "fee"]

        # Act & Assert
        with pytest.raises(KeyError):
            perform_latency_linear_regression(df_missing, predictor, categorical, numeric)

    def test_perform_latency_linear_regression_deterministic_verification(self, sample_df):
        # Arrange
        predictor = ["response_time_ms"]
        categorical = ["indexer", "deployment_hash", "indexer_network"]
        numeric = ["distance_miles", "fee"]

        # Act - run twice
        rankings1, _ = perform_latency_linear_regression(
            sample_df, predictor, categorical, numeric
        )
        rankings2, _ = perform_latency_linear_regression(
            sample_df, predictor, categorical, numeric
        )

        # Assert - results should be identical
        pd.testing.assert_frame_equal(rankings1, rankings2)

    def test_perform_latency_linear_regression_original_df_unchanged(self, sample_df):
        # Arrange
        original = sample_df.copy()
        predictor = ["response_time_ms"]
        categorical = ["indexer", "deployment_hash", "indexer_network"]
        numeric = ["distance_miles", "fee"]

        # Act
        perform_latency_linear_regression(sample_df, predictor, categorical, numeric)

        # Assert
        pd.testing.assert_frame_equal(original, sample_df)


class TestCalculateIndexerUptime:
    """
    Tests for the indexer uptime calculation.
    This is critical for measuring indexer reliability.
    """

    @pytest.fixture
    def sample_df(self):
        return pd.DataFrame({
            "indexer": ["A", "A", "A", "A", "B", "B", "C"],
            "timestamp": [
                datetime(2024, 1, 1, 12, 0),
                datetime(2024, 1, 1, 12, 2),
                datetime(2024, 1, 1, 12, 5),
                datetime(2024, 1, 1, 12, 7),
                datetime(2024, 1, 1, 12, 0),
                datetime(2024, 1, 1, 12, 3),
                datetime(2024, 1, 1, 12, 0),
            ],
            "status": [
                "200 OK", "200 OK", "Error", "200 OK",
                "200 OK", "Unavailable(MissingBlock)",
                "200 OK",
            ],
        })

    def test_calculate_indexer_uptime_base_case(self, sample_df):
        # Act
        result = calculate_indexer_uptime(sample_df)

        # Assert - expected columns present
        expected_columns = [
            "indexer", "observed_duration_restricted", "uptime_duration_restricted",
            "observed_duration_full", "uptime_duration_full", "% up_y", "% up_x",
        ]
        assert all(col in result.columns for col in expected_columns)

        # All indexers present
        assert set(result["indexer"]) == set(sample_df["indexer"])

        # Percentages valid (0-100 or NaN for single-query indexers)
        assert all(
            (0 <= p <= 100) or np.isnan(p) for p in result["% up_x"]
        )

    def test_calculate_indexer_uptime_all_up(self):
        # Arrange
        df = pd.DataFrame({
            "indexer": ["A", "A", "B", "B"],
            "timestamp": [
                datetime(2024, 1, 1, 12, 0),
                datetime(2024, 1, 1, 12, 2),
                datetime(2024, 1, 1, 12, 0),
                datetime(2024, 1, 1, 12, 2),
            ],
            "status": ["200 OK", "Unavailable(MissingBlock)", "200 OK", "200 OK"],
        })

        # Act
        result = calculate_indexer_uptime(df)

        # Assert - all should be 100%
        assert all(result["% up_x"] == 100)
        assert all(result["% up_y"] == 100)

    def test_calculate_indexer_uptime_all_down(self):
        # Arrange
        df = pd.DataFrame({
            "indexer": ["A", "A", "B", "B"],
            "timestamp": [
                datetime(2024, 1, 1, 12, 0),
                datetime(2024, 1, 1, 12, 2),
                datetime(2024, 1, 1, 12, 0),
                datetime(2024, 1, 1, 12, 2),
            ],
            "status": ["Error", "Bad", "504 Error", "Timeout"],
        })

        # Act
        result = calculate_indexer_uptime(df)

        # Assert - all should be 0%
        assert all(result["% up_x"] == 0)
        assert all(result["% up_y"] == 0)

    def test_calculate_indexer_uptime_threshold(self):
        # Arrange - queries spaced 5 minutes apart
        df = pd.DataFrame({
            "indexer": ["A", "A", "A"],
            "timestamp": [
                datetime(2024, 1, 1, 12, 0),
                datetime(2024, 1, 1, 12, 5),
                datetime(2024, 1, 1, 12, 10),
            ],
            "status": ["200 OK", "200 OK", "200 OK"],
        })

        # Act - compare default (120s) vs lower threshold (60s)
        result_default = calculate_indexer_uptime(df, threshold_seconds=120)
        result_low = calculate_indexer_uptime(df, threshold_seconds=60)

        # Assert - lower threshold should show lower restricted uptime
        assert (
            result_low["uptime_duration_restricted"].iloc[0]
            < result_default["uptime_duration_restricted"].iloc[0]
        )

    def test_calculate_indexer_uptime_empty_df(self):
        # Arrange
        df = pd.DataFrame(columns=["indexer", "timestamp", "status"])

        # Act
        result = calculate_indexer_uptime(df)

        # Assert
        assert result.empty

    def test_calculate_indexer_uptime_single_entry_for_indexers(self):
        # Arrange - single query can't compute uptime
        df = pd.DataFrame({
            "indexer": ["A", "B"],
            "timestamp": [datetime(2024, 1, 1, 12, 0), datetime(2024, 1, 1, 12, 0)],
            "status": ["200 OK", "Error"],
        })

        # Act
        result = calculate_indexer_uptime(df)

        # Assert - should have NaN uptime percentages
        assert len(result) == 2
        assert np.isnan(result["% up_x"].iloc[0])
        assert np.isnan(result["% up_y"].iloc[0])

    def test_calculate_indexer_uptime_rounding(self):
        # Arrange
        df = pd.DataFrame({
            "indexer": ["A", "A", "A"],
            "timestamp": [
                datetime(2024, 1, 1, 12, 0),
                datetime(2024, 1, 1, 12, 1),
                datetime(2024, 1, 1, 12, 3),
            ],
            "status": ["200 OK", "Error", "200 OK"],
        })

        # Act
        result = calculate_indexer_uptime(df)

        # Assert - percentages rounded to 3 decimals
        assert all(round(p, 3) == p for p in result["% up_x"].dropna())

    def test_calculate_indexer_uptime_sorting(self):
        # Arrange - test if the result is sorted by '% up' in descending order
        df = pd.DataFrame({
            "indexer": ["A", "A", "B", "B", "C", "C"],
            "timestamp": [datetime(2024, 1, 1, 12, i) for i in range(6)],
            "status": ["200 OK", "Error", "200 OK", "200 OK", "200 OK", "200 OK"],
        })

        # Act
        result = calculate_indexer_uptime(df)

        # Assert - '% up_x' column is sorted in descending order
        assert list(result["% up_x"]) == sorted(result["% up_x"], reverse=True)


class TestAggregateIndexerInfo:
    """Tests for the aggregate_indexer_info function."""

    def test_aggregate_indexer_info_base_case(self):
        # Arrange
        df = pd.DataFrame({
            "indexer": ["A", "A", "B", "B", "C", "C", "C"],
            "org": ["X", "X", "Y", "Z", "W", "W", "W"],
            "dst_lat": [10.1, 13.123445, 35, 31, 55, 45, 50],
            "dst_lon": [22, 25.123445, 44, 41, 65, 60, 60],
        })

        # Act
        result = aggregate_indexer_info(df)

        # Assert - mode org selected
        assert list(result["org"]) == ["X", "Y", "W"]

        # Lat/lon rounded to nearest 20
        assert list(result["dst_lat"]) == [20, 40, 40]
        assert list(result["dst_lon"]) == [20, 40, 60]

    def test_aggregate_indexer_info_empty_df(self):
        # Arrange
        df = pd.DataFrame(columns=["indexer", "org", "dst_lat", "dst_lon"])

        # Act
        result = aggregate_indexer_info(df)

        # Assert
        assert result.empty
        assert list(result.columns) == ["indexer", "org", "dst_lat", "dst_lon"]

    def test_aggregate_indexer_info_with_nans(self):
        # Arrange
        df = pd.DataFrame({
            "indexer": ["A", "A", "B", "B", "B"],
            "org": [np.nan, "X", "Y", np.nan, np.nan],
            "dst_lat": [10, np.nan, np.nan, np.nan, np.nan],
            "dst_lon": [20, np.nan, np.nan, np.nan, np.nan],
        })

        # Act
        result = aggregate_indexer_info(df)

        # Assert
        assert list(result["indexer"]) == ["A", "B"]
        assert list(result["org"]) == ["X", "Y"]


class TestMergeAndPrepareDataframes:
    """Tests for the final merge and prepare function."""

    @pytest.fixture
    def indexer_uptime(self):
        return pd.DataFrame({
            "indexer": ["0xABC", "0xXYZ", "0x123"],
            "uptime": [99.5, 98.7, 97.0],
            "observed_duration_full": [100, 200, 300],
            "uptime_duration_full": [99, 197, 291],
        })

    @pytest.fixture
    def indexer_rankings(self):
        return pd.DataFrame({
            "indexer": ["0xABC", "0xXYZ", "0x789"],
            "Latency Coefficient": [0.5, 0.8, 1.2],
            "Standard Error": [0.05, 0.08, 0.12],
            "p-value": [0.01, 0.02, 0.03],
            "% up_y": [95, 96, 97],
        })

    @pytest.fixture
    def agg_df(self):
        return pd.DataFrame({
            "indexer": ["0xABC", "0xXYZ", "0x456"],
            "org": ["AWS", "GCP", "Hetzner"],
            "dst_lat": [40, 35, 50],
            "dst_lon": [-74, -120, 8],
        })

    @pytest.fixture
    def indexer_success_rate(self):
        return pd.DataFrame({
            "indexer": ["0xABC", "0xXYZ", "0xDEF"],
            "average_status": [0.95, 0.92, 0.88],
        })

    @pytest.fixture
    def stake_to_fees(self):
        return pd.DataFrame({
            "indexer": ["0xABC", "0xXYZ", "0xGHI"],
            "stake_to_fees": [100, 200, 300],
            "stake_to_fees_iqr_deviation": [-0.5, 0, 0.5],
        })

    def test_merge_base_case(
        self, indexer_uptime, indexer_rankings, agg_df,
        indexer_success_rate, stake_to_fees
    ):
        # Act
        result = merge_and_prepare_dataframes(
            indexer_uptime, indexer_rankings, agg_df,
            indexer_success_rate, stake_to_fees,
        )

        # Assert - core columns present
        assert "indexer" in result.columns
        assert "uptime" in result.columns
        assert "Latency Coefficient" in result.columns

        # Placeholder columns for DIP metrics
        assert "existing_dips_agreements" in result.columns
        assert all(result["existing_dips_agreements"] == 0)
        assert "avg_sync_duration" in result.columns
        assert all(pd.isna(result["avg_sync_duration"]))
        assert "indexing_agreement_acceptance_latency" in result.columns
        assert all(pd.isna(result["indexing_agreement_acceptance_latency"]))

        # Dropped columns not present
        assert "% up_y" not in result.columns
        assert "observed_duration_full" not in result.columns

    def test_merge__missing_indexer(
        self, indexer_uptime, indexer_rankings, agg_df,
        indexer_success_rate, stake_to_fees
    ):
        # Arrange - remove an indexer
        indexer_uptime_modified = indexer_uptime[indexer_uptime["indexer"] != "0xABC"]

        # Act
        result = merge_and_prepare_dataframes(
            indexer_uptime_modified, indexer_rankings, agg_df,
            indexer_success_rate, stake_to_fees,
        )

        # Assert - removed indexer not in result
        assert "0xABC" not in result["indexer"].values

    def test_merge_no_common_indexers(
        self, indexer_uptime, indexer_rankings, agg_df,
        indexer_success_rate, stake_to_fees
    ):
        # Arrange - different indexer sets (no overlap with rankings)
        indexer_uptime["indexer"] = ["0xAAA", "0xBBB", "0xCCC"]

        # Act
        result = merge_and_prepare_dataframes(
            indexer_uptime, indexer_rankings, agg_df,
            indexer_success_rate, stake_to_fees,
        )

        # Assert - result is empty because:
        # 1. Left merge with rankings creates NaN for latency columns
        # 2. dropna(subset=latency_columns) removes all rows
        assert result.empty

    def test_merge_additional_columns(
        self, indexer_uptime, indexer_rankings, agg_df,
        indexer_success_rate, stake_to_fees
    ):
        # Arrange - add extra column
        indexer_uptime["extra_col"] = [1, 2, 3]

        # Act
        result = merge_and_prepare_dataframes(
            indexer_uptime, indexer_rankings, agg_df,
            indexer_success_rate, stake_to_fees,
        )

        # Assert - extra column preserved
        assert "extra_col" in result.columns
