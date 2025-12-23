"""
Test suite covering the DataManager class.
"""

from datetime import datetime
from typing import Optional, Tuple
from unittest.mock import MagicMock, patch

import numpy as np
import pandas as pd
import pytest
from sklearn.compose import ColumnTransformer
from sklearn.pipeline import Pipeline

from __fixtures__ import network as network_fixture
from iisa.data_manager.manager import (
    DataManager,
    IndexerRankingsDataFrame,
    IndexerRankingsSchema,
    LinearRegressionResultsSchema,
)
from iisa.data_manager.processing import (
    _CalculateDistancesInputDataFrame,
    _CalculateDistancesInputSchema,
    _CalculateDistancesMixinSchema,
    _FilterSuccessfulQueriesInputDataFrame,
    _FilterSuccessfulQueriesInputSchema,
    _FilterSuccessfulQueriesMixinSchema,
    _MergeInIndexersInfoInputDataFrame,
    _MergeInIndexersInfoInputSchema,
    _MergeInIndexersInfoMixinSchema,
    _MergeInQueryGeolocationInputDataFrame,
    _MergeInQueryGeolocationInputSchema,
    _MergeInQueryGeolocationMixinSchema,
    _latency_linear_regression_analyze_results,
    _latency_linear_regression_calculate_robust_normalized_coefficients,
    _latency_linear_regression_create_pipeline,
    _latency_linear_regression_preprocess_data,
    adjust_rows,
    aggregate_indexer_info,
    calculate_distances,
    calculate_indexer_stake_to_fees,
    calculate_indexer_success_rate,
    calculate_indexer_uptime,
    filter_successful_queries,
    hash_sampled_queries,
    iterative_filter,
    merge_and_prepare_dataframes,
    merge_in_indexers_info,
    merge_in_query_geolocation_info,
    perform_latency_linear_regression,
    strategic_sample,
)
from iisa.geoip import GeoipResolver
from iisa.network import IndexersDataFrame, IndexersSchema, NetworkProvider
from iisa.perf import (
    PerfHistoryDataFrame,
    PerfHistorySchema,
)
from iisa.typing import empty_dataframe


@pytest.fixture
def sample_data():
    return pd.DataFrame(
        {
            "indexer": ["A", "B", "C"],
            "deployment_hash": ["hash1", "hash2", "hash3"],
            "score": [0.8, 0.6, 0.7],
        }
    )


@pytest.fixture
def mock__regression_results():
    filtered_df = pd.DataFrame(
        {
            "indexer": ["indexer1", "indexer2", "indexer3"],
            "coefficient": [0.1, 0.2, 0.3],
            "p_value": [0.01, 0.02, 0.03],
        }
    )
    rankings_df = pd.DataFrame(
        {"indexer": ["indexer1", "indexer2", "indexer3"], "rank": [1, 2, 3]}
    )
    return filtered_df, rankings_df


@pytest.fixture
def mock__combined_query_results(faker):
    return pd.DataFrame(
        {
            "query_id": [faker.query_id() for _ in range(3)],
            "deployment_hash": [faker.deployment_id() for _ in range(3)],
            "indexer": [faker.indexer_id() for _ in range(3)],
            "indexer_network": ["net1", "net2", "net3"],
            "org": ["hetzner", "amazon aws", "google"],
            "fee": [0.1, 0.2, 0.3],
            "timestamp": ["2024-01-01", "2024-01-02", "2024-01-03"],
            "blocks_behind": [1, 2, 3],
            "response_time_ms": [100, 200, 300],
            "status": ["200 OK", "200 OK", "200 OK"],
            "day_partition": ["2024-01-01", "2024-01-02", "2024-01-03"],
            "subgraph_network": ["network1", "network2", "network3"],
            "url": [faker.url() for _ in range(3)],
            "origin_loc": ["0,20", "40,40", "60,60"],
            "destination_loc": ["20,40", "40,60", "60,80"],
            "loc": ["0,20", "40,40", "60,60"],
            "distance_miles": [100, 200, 300],
            "sampled_query_id_hashed_mod_integer_root": [0, 1, 2],
        }
    )


@pytest.fixture
def mock__bigquery_provider(faker, mock__combined_query_results):
    bigquery_provider = MagicMock()
    bigquery_provider.return_value.fetch_initial_query_results.return_value = (
        pd.DataFrame(
            {
                "deployment_hash": [faker.deployment_id() for _ in range(3)],
                "indexer": [faker.indexer_id() for _ in range(3)],
                "num_rows": [1000, 2000, 3000],
            }
        )
    )
    bigquery_provider.return_value.fetch_combined_query_results.return_value = (
        mock__combined_query_results
    )
    bigquery_provider.return_value.fetch_initial_stake_to_fees.return_value = (
        pd.DataFrame(
            {
                "indexer": [faker.indexer_id() for _ in range(3)],
                "stake_to_fees": [1.0, 2.0, 3.0],
            }
        )
    )
    return bigquery_provider


@pytest.fixture
def mock__network_provider(ipinfo_io_auth):
    resolver = GeoipResolver(ipinfo_io_auth)
    provider = NetworkProvider(geoip=resolver)

    # Initialize the network provider with test data
    test_data = network_fixture.load_fixture_data()
    provider.set_snapshot(test_data)

    return provider


@pytest.mark.skip(reason="requires a new IPinfo.io API key")
class TestDataManagerClass:
    """
    This class contains tests to ensure that the DataManager class
    correctly initializes, fetches data, and provides access to its data.
    """

    def test_initialize_data_manager(
        self, mock__bigquery_provider, mock__network_provider
    ):
        ## Given
        bigquery_provider = mock__bigquery_provider.return_value
        network_provider = mock__network_provider

        ## When
        result = DataManager(bigquery_provider, network_provider)

        ## Then
        # Verify the num_days and end_date attributes
        assert isinstance(result.num_days, int)
        assert isinstance(result.end_date, Optional[datetime])

        # Verify data outputs are None
        assert result.get_data() is None
        assert result.get_latency_linear_regression_indexer_rankings() is None
        assert result.get_latency_linear_regression_results() is None

    def test_fetch_and_update(
        self, faker, mock__bigquery_provider, mock__network_provider
    ):
        ## Given
        bigquery_provider = mock__bigquery_provider.return_value
        network_provider = mock__network_provider

        def mock__fetch_and_process_data(
            *args, **kwargs
        ) -> Tuple[pd.DataFrame, pd.DataFrame, pd.DataFrame]:
            """
            Creates and returns mock data simulating BigQuery fetch results.

            This function generates mock data for:
            1. query_data: A DataFrame with various indexer and query metrics.
            3. indexer_rankings: A DataFrame with indexer rankings and scores.
            """
            request_data = PerfHistoryDataFrame(
                {
                    "query_id": [faker.query_id() for _ in range(3)],
                    "deployment_hash": [faker.deployment_id() for _ in range(3)],
                    "indexer": [faker.indexer_id() for _ in range(3)],
                    "url": [faker.url() for _ in range(3)],
                }
            )
            latency_linear_regression_indexer_rankings = IndexerRankingsDataFrame(
                {
                    "indexer": [faker.indexer_id() for _ in range(3)],
                    "rank": [1, 2, 3],
                    "score": [0.9, 0.8, 0.7],
                }
            )
            latency_linear_regression_results_df_mock_data = pd.DataFrame(
                {
                    "variable": ["var1", "var2", "var3"],
                    "coefficient": [0.1, 0.2, 0.3],
                    "standard_error": [0.01, 0.02, 0.03],
                    "p_value": [0.001, 0.002, 0.003],
                }
            )

            return (
                request_data,
                latency_linear_regression_indexer_rankings,
                latency_linear_regression_results_df_mock_data,
            )

        data_manager = DataManager(bigquery_provider, network_provider)

        ## When
        # Patch internal implementation for the test
        with patch(
            "iisa.data_manager.manager._fetch_and_process_data",
            mock__fetch_and_process_data,
        ):
            data_manager.fetch_data_and_update()

        ## Then
        data = data_manager.get_data()
        indexer_rankings = data_manager.get_latency_linear_regression_indexer_rankings()
        regression_results = data_manager.get_latency_linear_regression_results()

        # Verify data is present and non-empty, and that it conforms to the Pandera schema
        assert data is not None
        assert not data.empty
        PerfHistorySchema.validate(data)

        assert indexer_rankings is not None
        assert not indexer_rankings.empty
        IndexerRankingsSchema.validate(indexer_rankings)

        assert regression_results is not None
        assert not regression_results.empty
        LinearRegressionResultsSchema.validate(regression_results)

    def test_fetch_and_update_failure(
        self, mock__bigquery_provider, mock__network_provider
    ):
        """
        This test verifies that initialize_data_manager handles exceptions gracefully.
        """
        ## Given
        # Set up the mock to raise an exception when fetching BigQuery data
        mock__bigquery_provider.return_value.fetch_initial_query_results.side_effect = (
            RuntimeError("Mock error")
        )

        bigquery_provider = mock__bigquery_provider.return_value
        network_provider = mock__network_provider

        data_manager = DataManager(bigquery_provider, network_provider)

        ## When
        # Verify the function raises the expected exception
        with pytest.raises(RuntimeError) as ex:
            data_manager.fetch_data_and_update()

        ## Then
        # Assert the exception message
        assert str(ex.value) == "Mock error"

    def test_get_data(self, mock__bigquery_provider, mock__network_provider):
        ## Given
        bigquery_provider = mock__bigquery_provider.return_value
        network_provider = mock__network_provider

        # Initialize a DataManager instance
        data_manager = DataManager(bigquery_provider, network_provider)

        # Mock the `_fetch_and_process_data` method to avoid actual data fetching
        mock__data = pd.DataFrame(
            {
                "indexer": ["indexer1", "indexer2", "indexer3"],
                "score": [0.9, 0.8, 0.7],
                "query_count": [100, 200, 300],
            }
        )
        with patch(
            "iisa.data_manager.manager._fetch_and_process_data",
            return_value=(mock__data, None, None),
        ):
            data_manager.fetch_data_and_update()

        ## When
        result = data_manager.get_data()

        ## Then
        # Verify returned data is the same as the mock data
        pd.testing.assert_frame_equal(result, mock__data)

    def test_get_latency_linear_regression_indexer_rankings(
        self, mock__bigquery_provider, mock__network_provider
    ):
        ## Given
        bigquery_provider = mock__bigquery_provider.return_value
        network_provider = mock__network_provider

        # Initialize a DataManager instance
        data_manager = DataManager(bigquery_provider, network_provider)

        # Mock the `_fetch_and_process_data` method to avoid actual data fetching
        mock__data = pd.DataFrame({"column1": [1, 2, 3]})
        mock__indexer_rankings = pd.DataFrame(
            {"indexer": ["A", "B", "C"], "rank": [1, 2, 3]}
        )
        mock__regression_results = pd.DataFrame({"result": [4, 5, 6]})
        with patch(
            "iisa.data_manager.manager._fetch_and_process_data",
            return_value=(
                mock__data,
                mock__indexer_rankings,
                mock__regression_results,
            ),
        ):
            data_manager.fetch_data_and_update()

        ## When
        indexer_rankings = data_manager.get_latency_linear_regression_indexer_rankings()
        regression_results = data_manager.get_latency_linear_regression_results()

        # Verify returned data is the same as the mock data.
        pd.testing.assert_frame_equal(indexer_rankings, mock__indexer_rankings)
        pd.testing.assert_frame_equal(regression_results, mock__regression_results)


class TestAdjustRows:
    """
    Tests for the adjust_rows function.

    This class tests various scenarios for adjusting the number of rows
    in a DataFrame to approximate a target total number of rows.
    """

    def test_adjust_rows_normal_case(self):
        # Setup sample data
        sample_data = pd.DataFrame(
            {
                "deployment_hash": ["hash1", "hash2", "hash3", "hash1"],
                "indexer": ["index1", "index2", "index3", "indexer4"],
                "num_rows": [50, 10000, 600, 50],
            }
        )

        # Test if adjustments approximate the target within the specified tolerance.
        target_rows = 600
        adjust_rows(sample_data, target_rows)
        adjusted_sum = sample_data["num_rows_restricted"].sum()
        assert target_rows * 0.99 <= adjusted_sum <= target_rows * 1.01

    def test_adjust_rows_empty_dataframe(self):
        # Setup an empty DataFrame
        df = pd.DataFrame({"deployment_hash": [], "indexer": [], "num_rows": []})

        # Test handling of empty data
        target_rows = 100
        adjust_rows(df, target_rows)
        assert df.empty

    def test_adjust_rows_zero_target(self):
        # Setup sample data
        sample_data = pd.DataFrame(
            {
                "deployment_hash": ["hash1", "hash2", "hash3", "hash1"],
                "indexer": ["index1", "index2", "index3", "indexer4"],
                "num_rows": [50, 10000, 600, 50],
            }
        )

        # Test response when the target number of rows is zero
        target_rows = 0
        adjust_rows(sample_data, target_rows)
        assert sample_data["num_rows_restricted"].sum() == 0

    def test_adjust_rows_negative_case(self):
        # Setup sample data with uniform distribution
        df = pd.DataFrame(
            {
                "deployment_hash": ["hash1", "hash1", "hash1", "hash1"],
                "indexer": ["index1", "index1", "index1", "index1"],
                "num_rows": [100, 100, 100, 100],
            }
        )

        # Test handling of negative target rows
        target_rows = -300
        with pytest.raises(
            ValueError, match="Target rows must be a non-negative integer"
        ):
            adjust_rows(df, target_rows)


class TestMergeInIndexersDataFrame:
    @pytest.fixture
    def combined_query_pandas(self):
        return _MergeInIndexersInfoInputDataFrame(
            {
                "indexer": [
                    "0x123fffffffffffffffffffffffffffffffffffff",
                    "0x456fffffffffffffffffffffffffffffffffffff",
                    "0x789fffffffffffffffffffffffffffffffffffff",
                ],
                "url": [
                    "https://example.com",
                    "https://test.com",
                    "https://another.com",
                ],
            }
        )

    @pytest.fixture
    def indexers(self):
        return IndexersDataFrame(
            {
                "indexer": [
                    "0x123fffffffffffffffffffffffffffffffffffff",
                    "0x456fffffffffffffffffffffffffffffffffffff",
                ],
                "url": ["https://example.com", "https://test.com"],
                "indexer_network": ["arbitrum", "arbitrum"],
                "ip_addr": ["1.1.2.2", "3.3.4.4"],
                "org": ["Org1", "Org2"],
                "country": ["US", "CN"],
                "latitude": [1.0, 2.0],
                "longitude": [1.0, 2.0],
            }
        )

    def test_merge_in_indexers_info(self, combined_query_pandas, indexers):
        ## When
        result = merge_in_indexers_info(combined_query_pandas, indexers)

        ## Then
        # Assert the result complies with the input and output schemas
        _MergeInIndexersInfoInputSchema.validate(result)
        _MergeInIndexersInfoMixinSchema.validate(result)

        # Assert the result content
        expected = pd.DataFrame(
            {
                "indexer": [
                    "0x123fffffffffffffffffffffffffffffffffffff",
                    "0x456fffffffffffffffffffffffffffffffffffff",
                    "0x789fffffffffffffffffffffffffffffffffffff",
                ],
                "url": [
                    "https://example.com",
                    "https://test.com",
                    "https://another.com",
                ],
                "indexer_network": ["arbitrum", "arbitrum", None],
                "ip_addr": ["1.1.2.2", "3.3.4.4", None],
                "org": ["Org1", "Org2", None],
                "dst_country": ["US", "CN", None],
                "dst_lat": [1.0, 2.0, None],
                "dst_lon": [1.0, 2.0, None],
            }
        )

        pd.testing.assert_frame_equal(result, expected)

    def test_merge_into_empty_combined_queries_results(self, indexers):
        ## Given
        data = empty_dataframe(_MergeInIndexersInfoInputSchema)

        ## When
        result = merge_in_indexers_info(data, indexers)

        ## Then
        # Assert the result complies with the input and output schemas
        _MergeInIndexersInfoInputSchema.validate(result)
        _MergeInIndexersInfoMixinSchema.validate(result)

        # Assert that the result is an empty dataframe
        assert result.empty

    def test_merge_in_empty_indexers_info(self, combined_query_pandas):
        ## Given
        indexers_df = empty_dataframe(IndexersSchema)

        ## When
        result = merge_in_indexers_info(combined_query_pandas, indexers_df)

        ## Then
        # Assert the result complies with the input and output schemas
        _MergeInIndexersInfoInputSchema.validate(result)
        _MergeInIndexersInfoMixinSchema.validate(result)

        # Assert the result is non-empty and the new columns are filled with NaN values
        assert not result.empty

        assert result["indexer_network"].isna().all()
        assert result["ip_addr"].isna().all()
        assert result["org"].isna().all()
        assert result["dst_country"].isna().all()
        assert result["dst_lat"].isna().all()
        assert result["dst_lon"].isna().all()


class TestMergeInQueryGeolocationInfo:
    def test_merge_in_iata_info(self):
        ## Given
        data = _MergeInQueryGeolocationInputDataFrame(
            {
                "query_id": [
                    "1111111111111111-AMS",
                    "2222222222222222-CDG",
                    "3333333333333333-LHR",
                ],
            }
        )

        ## When
        result = merge_in_query_geolocation_info(data)

        ## Then
        # Assert the result complies with the input and output schemas
        _MergeInQueryGeolocationInputSchema.validate(result)
        _MergeInQueryGeolocationMixinSchema.validate(result)

        # Assert the result content
        expected = pd.DataFrame(
            {
                "query_id": [
                    "1111111111111111-AMS",
                    "2222222222222222-CDG",
                    "3333333333333333-LHR",
                ],
                "IATA_code": ["AMS", "CDG", "LHR"],
                "src_country": ["NL", "FR", "GB"],
                "src_lat": [52.3086, 49.0128, 51.4706],
                "src_lon": [4.7639, 2.5500, -0.46194],
            }
        )
        pd.testing.assert_frame_equal(result, expected)

    def test_merge_with_unknown_iata_code(self):
        ## Given
        data = _MergeInQueryGeolocationInputDataFrame(
            {
                "query_id": [
                    "1111111111111111-AMS",
                    "2222222222222222-CDG",
                    "3333333333333333-LHR",
                    "0000000000000000-XXX",
                ],
            }
        )

        ## When
        result = merge_in_query_geolocation_info(data)

        ## Then
        # Assert the result complies with the input and output schemas
        _MergeInQueryGeolocationInputSchema.validate(result)
        _MergeInQueryGeolocationMixinSchema.validate(result)

        # Assert the result content
        assert result.loc[3, "IATA_code"] == "XXX"
        assert pd.isna(result.loc[3, "src_country"])
        assert pd.isna(result.loc[3, "src_lat"])
        assert pd.isna(result.loc[3, "src_lon"])

    def test_merge_with_empty_dataframe(self):
        ## Given
        data = empty_dataframe(_MergeInQueryGeolocationInputSchema)

        ## When
        result = merge_in_query_geolocation_info(data)

        ## Then
        # Assert the result complies with the input and output schemas
        _MergeInQueryGeolocationInputSchema.validate(result)
        _MergeInQueryGeolocationMixinSchema.validate(result)

        # Assert result is as expected.
        assert result.empty


class TestCalculateDistances:
    @pytest.fixture
    def sample_df(self):
        return _CalculateDistancesInputDataFrame(
            {
                "src_lon": [-74.4444, -118.8888, -0.3333],
                "src_lat": [40.5555, 34.9999, 51.4444],
                "dst_lon": [-87.6666, -122.1111, 2.5555],
                "dst_lat": [41.7777, 37.2222, 48.6666],
            }
        )

    def test_calculate_distance(self):
        ## Given
        data = _CalculateDistancesInputDataFrame(
            {
                "src_lon": [0.0],
                "src_lat": [0.0],
                "dst_lon": [30.0],
                "dst_lat": [0.0],
            }
        )

        ## When
        result = calculate_distances(data)

        ## Then
        # Assert the result complies with the output schema
        _CalculateDistancesMixinSchema.validate(result)

        # Assert the result content
        expected_distance = 2000  # 2072.7 miles, approximate distance for 30 degrees of longitude at the equator
        assert result["distance_miles"].iloc[0] == expected_distance

    def test_calculate_distances_multiple(self, sample_df):
        ## When
        result = calculate_distances(sample_df)

        ## Then
        # Assert the result complies with the output schema
        _CalculateDistancesMixinSchema.validate(result)

        # Assert the result content
        assert len(sample_df) == len(result)
        assert result["distance_miles"].notna().all()

    def test_calculate_same_location(self):
        ## Given
        data = _CalculateDistancesInputDataFrame(
            {
                "src_lon": [10.0, 20.0],
                "src_lat": [10.0, 20.0],
                "dst_lon": [10.0, 20.0],
                "dst_lat": [10.0, 20.0],
            }
        )

        ## When
        result = calculate_distances(data)

        ## Then
        # Assert the result complies with the output schema
        _CalculateDistancesMixinSchema.validate(result)

        # Assert all distances are zero
        assert result["distance_miles"].eq(0.0).all()

    def test_calculate_distances_empty_df(self):
        ## Given
        data = empty_dataframe(_CalculateDistancesInputSchema)

        ## When
        result = calculate_distances(data)

        ## Then
        # Assert the result complies with the output schema
        _CalculateDistancesMixinSchema.validate(result)

        # Assert the result is an empty DataFrame
        assert result.empty

    def test_calculate_distances_nan_values(self):
        ## Given
        data = _CalculateDistancesInputDataFrame(
            {
                "src_lat": [40.99, None, 51.20, 37.25],
                "src_lon": [-74.00, -118.00, None, -122.00],
                "dst_lat": [40.99, 37.25, 48.10, None],
                "dst_lon": [-84.00, None, 2.20, 2.20],
            }
        )

        ## When
        result = calculate_distances(data)

        ## Then
        # Assert the result complies with the output schema
        _CalculateDistancesMixinSchema.validate(result)

        # Assert the result's first row has a non-zero distance
        assert result["distance_miles"].iloc[0] >= 0

        # Assert the rest of the distances are NaN
        assert pd.isna(result["distance_miles"].iloc[1])
        assert pd.isna(result["distance_miles"].iloc[2])
        assert pd.isna(result["distance_miles"].iloc[3])


class TestFilterStatus:
    def test_filter_200_ok_status(self):
        ## Given
        data = _FilterSuccessfulQueriesInputDataFrame(
            {
                "status": [
                    "200 OK",
                    "404 Not Found",
                    "200 OK",
                    "500 Internal Server Error",
                    "200 OK",
                ],
                "data": ["A", "B", "C", "D", "E"],
            }
        )

        ## When
        result = filter_successful_queries(data)

        ## Then
        # Assert the result complies with the output schema
        _FilterSuccessfulQueriesMixinSchema.validate(result)

        # Assert the result content
        assert len(result) == 3
        assert list(result["data"]) == ["A", "C", "E"]
        assert result["status"].eq("200 OK").all()

    def test_filter_status_empty_df(self):
        ## Given
        data = empty_dataframe(_FilterSuccessfulQueriesInputSchema)

        ## When
        result = filter_successful_queries(data)

        ## Then
        # Assert the result complies with the output schema
        _FilterSuccessfulQueriesMixinSchema.validate(result)

        # Assert the result is empty
        assert result.empty

    def test_filter_status_with_nan_values(self):
        ## Given
        data = _FilterSuccessfulQueriesInputDataFrame(
            {
                "status": ["200 OK", pd.NA, "200 OK", None],
                "data": ["A", "B", "C", "D"],
            }
        )

        ## When
        result = filter_successful_queries(data)

        ## Then
        # Assert the result complies with the output schema
        _FilterSuccessfulQueriesMixinSchema.validate(result)

        # Assert the result content
        assert len(result) == 2
        assert list(result["data"]) == ["A", "C"]
        assert result["status"].eq("200 OK").all()


class TestIterativeFilter:
    @pytest.fixture
    def sample_df(self):
        data = {
            "deployment_hash": ["A"] * 12 + ["B"] * 8 + ["C"] * 8 + ["D"] * 4,
            "indexer": (["X", "Y", "Z"] * 10 + ["X", "Y"])[:32],
            "query_id": list(range(1, 33)),
        }
        return pd.DataFrame(data)

    def test_iterative_filter_base_case(self, sample_df):
        result = iterative_filter(
            sample_df,
            min_deployment_indexers=2,
            min_deployments_per_indexer=2,
            min_queries_per_indexer=2,
            min_queries_per_deployment=2,
        )
        assert len(result) == 32
        assert result["deployment_hash"].value_counts().to_dict() == {
            "A": 12,
            "B": 8,
            "C": 8,
            "D": 4,
        }
        assert result["indexer"].value_counts().to_dict() == {"X": 11, "Y": 11, "Z": 10}
        assert len(result["query_id"].unique()) == 32

    def test_iterative_filter_no_change(self, sample_df):
        result = iterative_filter(
            sample_df,
            min_deployment_indexers=0,
            min_deployments_per_indexer=0,
            min_queries_per_indexer=0,
            min_queries_per_deployment=0,
        )
        pd.testing.assert_frame_equal(result, sample_df)

    def test_iterative_filter_empty_result(self, sample_df):
        result = iterative_filter(
            sample_df,
            min_deployment_indexers=100,
            min_deployments_per_indexer=100,
            min_queries_per_indexer=100,
            min_queries_per_deployment=100,
        )
        assert len(result) == 0

    def test_iterative_filter_indexers_per_deployment_only(self, sample_df):
        result = iterative_filter(
            sample_df,
            min_deployment_indexers=3,
            min_deployments_per_indexer=0,
            min_queries_per_indexer=0,
            min_queries_per_deployment=0,
        )
        assert len(result) == 32
        assert set(result["deployment_hash"]) == {"A", "B", "C", "D"}

    def test_iterative_filter_deployments_per_indexer_only(self, sample_df):
        result = iterative_filter(
            sample_df,
            min_deployment_indexers=0,
            min_deployments_per_indexer=4,
            min_queries_per_indexer=0,
            min_queries_per_deployment=0,
        )
        assert len(result) == 32
        assert set(result["indexer"]) == {"X", "Y", "Z"}

    def test_iterative_filter_queries_per_indexer_only(self, sample_df):
        result = iterative_filter(
            sample_df,
            min_deployment_indexers=0,
            min_deployments_per_indexer=0,
            min_queries_per_indexer=11,
            min_queries_per_deployment=0,
        )
        assert len(result) == 22
        assert set(result["indexer"]) == {"X", "Y"}

    def test_iterative_filter_queries_per_deployment_only(self, sample_df):
        result = iterative_filter(
            sample_df,
            min_deployment_indexers=0,
            min_deployments_per_indexer=0,
            min_queries_per_indexer=0,
            min_queries_per_deployment=10,
        )
        assert len(result) == 12
        assert set(result["deployment_hash"]) == {"A"}

    def test_iterative_filter_empty_dataframe(self):
        df = pd.DataFrame(columns=["deployment_hash", "indexer", "query_id"])
        result = iterative_filter(
            df,
            min_deployment_indexers=0,
            min_deployments_per_indexer=0,
            min_queries_per_indexer=0,
            min_queries_per_deployment=0,
        )
        assert len(result) == 0
        assert list(result.columns) == ["deployment_hash", "indexer", "query_id"]


class TestStrategicSample:
    @pytest.fixture
    def sample_df(self):
        return pd.DataFrame(
            {
                "deployment_hash": ["A", "A", "A", "B", "B", "C"] * 10000,
                "indexer": ["X", "Y", "Z", "X", "Y", "X"] * 10000,
                "query_id": range(60000),
            }
        )

    def test_strategic_sample_basic(self, sample_df):
        # Compute the result
        result_df, integer_root = strategic_sample(
            sample_df, target_rows_per_subgraph=30
        )

        # Sample the sampled_query_id's
        sampled = result_df[result_df["sampled_query_id"].notna()]

        # Check the length of the output df has not changed
        assert len(result_df) == len(sample_df)

        # Check the number of not none rows in the output df as as expected
        assert result_df["sampled_query_id"].notna().sum() == 90

        # Verify the integer root is the expected integer
        assert isinstance(integer_root, int)
        assert integer_root == int(np.sqrt(result_df["sampled_query_id"].notna().sum()))

        # Calculate the number of unique indexers per deployment_hash
        indexers_per_subgraph = sampled.groupby("deployment_hash")["indexer"].nunique()

        # Verify there is at least 1 indexer per subgraph
        assert indexers_per_subgraph.min() > 0

        # For this case verify the spread of the number of indexers serving sugraphs is exactly 2.
        assert (indexers_per_subgraph.max() - indexers_per_subgraph.min()) == 2

    def test_strategic_sample_empty_df(self):
        empty_df = pd.DataFrame(columns=["deployment_hash", "indexer", "query_id"])
        result_df, integer_root = strategic_sample(
            empty_df, target_rows_per_subgraph=10
        )

        assert result_df.empty
        assert "sampled_query_id" in result_df.columns
        assert integer_root == 0

    def test_strategic_sample_target_rows_per_subgraph_greater_than_df(self, sample_df):
        # Compute the result
        result_df, integer_root = strategic_sample(
            sample_df, target_rows_per_subgraph=10_000_000_000_000
        )

        # Check the length of the output df has not changed
        assert len(result_df) == len(sample_df)

        # Check that each query ID has been sampled exactly once. (since target_rows_per_subgraph > len(sample_df))
        assert (
            result_df["sampled_query_id"].notna().sum()
            == sample_df["query_id"].nunique()
        )


class TestHashSampledQueries:
    @pytest.fixture
    def sample_df(self):
        return pd.DataFrame(
            {
                "sampled_query_id": [1, 2, 3, None, 5, 6, np.nan, 8, 9, 10] * 1_000,
                "other_column": ["A", "B", "C", "D", "E", "F", "G", "H", "I", "J"]
                * 1_000,
            }
        )

    def test_hash_sampled_queries_basic(self, sample_df):
        integer_root = 33
        result = hash_sampled_queries(sample_df, integer_root)

        # Check that the new column is added
        assert "sampled_query_id_hashed_mod_integer_root" in result.columns

        # Check that only non-null sampled_query_id rows are hashed
        assert result["sampled_query_id_hashed_mod_integer_root"].notna().sum() == 8000
        assert result["sampled_query_id_hashed_mod_integer_root"].isna().sum() == 2000

        # Check that all hashed values are within the expected range
        assert all(
            0 <= x < integer_root
            for x in result["sampled_query_id_hashed_mod_integer_root"].dropna()
        )

    def test_hash_sampled_queries_empty_df(self):
        empty_df = pd.DataFrame(columns=["sampled_query_id"])
        result = hash_sampled_queries(empty_df, 5)
        assert "sampled_query_id_hashed_mod_integer_root" in result.columns
        assert result.empty

    def test_hash_sampled_queries_all_null(self):
        df = pd.DataFrame({"sampled_query_id": [None, None, None]})
        result = hash_sampled_queries(df, 5)
        assert "sampled_query_id_hashed_mod_integer_root" in result.columns
        assert result["sampled_query_id_hashed_mod_integer_root"].isna().all()

    def test_hash_sampled_queries_consistency(self, sample_df):
        integer_root = 7
        result1 = hash_sampled_queries(sample_df, integer_root)
        result2 = hash_sampled_queries(sample_df, integer_root)
        pd.testing.assert_frame_equal(result1, result2)

    def test_hash_sampled_queries_different_integer_roots(self, sample_df):
        result1 = hash_sampled_queries(sample_df.copy(), 5)
        result2 = hash_sampled_queries(sample_df.copy(), 10)

        assert not result1["sampled_query_id_hashed_mod_integer_root"].equals(
            result2["sampled_query_id_hashed_mod_integer_root"]
        )

    def test_hash_sampled_queries_original_df_unchanged(self, sample_df):
        original_df = sample_df.copy()
        _ = hash_sampled_queries(sample_df, 5)
        pd.testing.assert_frame_equal(original_df, sample_df)

    def test_hash_sampled_queries_large_integer_root(self):
        df = pd.DataFrame({"sampled_query_id": range(1000)})
        large_integer_root = 10_000_000_000
        result = hash_sampled_queries(df, large_integer_root)
        assert all(
            0 <= x < large_integer_root
            for x in result["sampled_query_id_hashed_mod_integer_root"]
        )


class TestPerformLinearRegression:
    """
    This integration test tests the perform_latency_linear_regression function and its dependencies:
    preprocess_data_for_latency_linear_regression, perform_latency_linear_regression, analyze_latency_linear_regression_results and
    calculate_robust_normalized_coefficients_latency_linear_regression.
    """

    @pytest.fixture
    def sample_df(self):
        # DataFrame with random data for testing
        np.random.seed(42)
        return pd.DataFrame(
            {
                "sampled_query_id": range(10_000),
                "indexer": np.random.choice(
                    ["0xABC", "0xXYZ", "0x123", "0x789"], 10_000
                ),
                "indexer_network": np.random.choice(
                    ["net1", "net2", "net3", "net4"], 10_000
                ),
                "deployment_hash": np.random.choice(
                    ["deployment_1", "deployment_2", "deployment_3", "deployment_4"],
                    10_000,
                ),
                "response_time_ms": np.random.randint(10, 20_000, 10_000),
                "fee": np.random.uniform(0.000001, 0.01, 10_000),
                "distance_miles": np.random.uniform(0, 1_000, 10_000),
                "score": np.random.uniform(0, 1, 10_000),
            }
        )

    def test_hash_sampled_queries_with_linear_regression(self, sample_df):
        # Apply hash_sampled_queries
        integer_root = 100
        hashed_df = hash_sampled_queries(sample_df, integer_root)

        # Check that the new column is added
        assert "sampled_query_id_hashed_mod_integer_root" in hashed_df.columns, (
            "sampled_query_id_hashed_mod_integer_root not in hash_df.columns"
        )

        # Setup linear regression variables
        predictor = ["response_time_ms"]
        categorical = [
            "indexer",
            "deployment_hash",
            "indexer_network",
            "sampled_query_id_hashed_mod_integer_root",
        ]
        numeric = ["distance_miles", "fee"]

        # Compute result
        (
            latency_linear_regression_indexer_rankings,
            latency_linear_regression_results_df,
        ) = perform_latency_linear_regression(
            hashed_df, predictor, categorical, numeric
        )

        # Check that latency_linear_regression_indexer_rankings contains expected columns
        expected_columns = [
            "indexer",
            "Latency Coefficient",
            "Standard Error",
            "p-value",
            "Latency Coefficient + Error Confidence Interval",
            "Robust Normalized Latency Coefficient + Error Confidence Interval",
        ]
        assert all(
            col in latency_linear_regression_indexer_rankings.columns
            for col in expected_columns
        ), "latency_linear_regression_results_df doesn't contain expected columns"

        # Check that only indexer values are present in the indexer column
        assert all(
            latency_linear_regression_indexer_rankings["indexer"].isin(
                ["0xABC", "0xXYZ", "0x123", "0x789"]
            )
        ), "indexer values, not present in the indexer column"

        # Check to ensure regression results are reasonable
        assert (
            latency_linear_regression_indexer_rankings["Latency Coefficient"]
            .notna()
            .all()
        )
        assert latency_linear_regression_indexer_rankings["p-value"].between(0, 1).all()

        # Check that the hashed column affects the regression by using a different mod hash integer root
        hashed_df_different_root = hash_sampled_queries(sample_df, integer_root + 1)
        (
            latency_linear_regression_indexer_rankings_different_root,
            latency_linear_regression_results_df_different_root,
        ) = perform_latency_linear_regression(
            hashed_df_different_root, predictor, categorical, numeric
        )

        assert not latency_linear_regression_indexer_rankings[
            "Latency Coefficient"
        ].equals(
            latency_linear_regression_indexer_rankings_different_root[
                "Latency Coefficient"
            ]
        )

    def test_preprocess_data_for_latency_linear_regression(self, sample_df):
        # Setup linear regression variables
        predictor = ["response_time_ms"]
        categorical = ["indexer", "deployment_hash", "indexer_network"]
        numeric = ["distance_miles", "fee"]

        # Perform preprocessing
        X, y, preprocessor = _latency_linear_regression_preprocess_data(
            sample_df, predictor, categorical, numeric
        )

        # Assert the correct types and structures of the preprocessed data
        assert isinstance(X, pd.DataFrame)
        assert isinstance(y, pd.DataFrame)
        assert isinstance(preprocessor, ColumnTransformer)
        assert list(y.columns) == predictor
        assert set(X.columns) == set(categorical + numeric)

    def test_perform_latency_linear_regression(self, sample_df):
        # Setup linear regression variables
        predictor = ["response_time_ms"]
        categorical = ["indexer", "deployment_hash", "indexer_network"]
        numeric = ["distance_miles", "fee"]

        # Preprocess data and perform regression
        X, y, preprocessor = _latency_linear_regression_preprocess_data(
            sample_df, predictor, categorical, numeric
        )
        pipeline, y_pred = _latency_linear_regression_create_pipeline(
            X, y, preprocessor
        )

        # Check the types and lengths of the regression outputs
        assert isinstance(pipeline, Pipeline)
        assert isinstance(y_pred, np.ndarray)
        assert len(y_pred) == len(y)

    def test_analyze_latency_linear_regression_results(self, sample_df):
        # Setup linear regression variables
        predictor = ["response_time_ms"]
        categorical = ["indexer", "deployment_hash", "indexer_network"]
        numeric = ["distance_miles", "fee"]

        # Perform regression and analyze results
        X, y, preprocessor = _latency_linear_regression_preprocess_data(
            sample_df, predictor, categorical, numeric
        )
        pipeline, y_pred = _latency_linear_regression_create_pipeline(
            X, y, preprocessor
        )
        results_df = _latency_linear_regression_analyze_results(pipeline, X, y, y_pred)

        # Check the structure and content of the results DataFrame
        assert isinstance(results_df, pd.DataFrame)
        assert set(results_df.columns) == {
            "Variable",
            "Latency Coefficient",
            "Standard Error",
            "p-value",
        }
        assert len(results_df) > 0

    def test_calculate_robust_normalized_coefficients_latency_linear_regression(
        self, sample_df
    ):
        # Setup linear regression variables
        predictor = ["response_time_ms"]
        categorical = ["indexer", "deployment_hash", "indexer_network"]
        numeric = ["distance_miles", "fee"]

        # Perform regression, analyze results, and calculate normalized coefficients
        X, y, preprocessor = _latency_linear_regression_preprocess_data(
            sample_df, predictor, categorical, numeric
        )
        pipeline, y_pred = _latency_linear_regression_create_pipeline(
            X, y, preprocessor
        )
        results_df = _latency_linear_regression_analyze_results(pipeline, X, y, y_pred)
        indexer_rankings = (
            _latency_linear_regression_calculate_robust_normalized_coefficients(
                results_df
            )
        )

        # Check the structure and content of the indexer rankings DataFrame
        assert isinstance(indexer_rankings, pd.DataFrame)
        assert set(indexer_rankings.columns) == {
            "indexer",
            "Latency Coefficient",
            "Standard Error",
            "p-value",
            "Latency Coefficient + Error Confidence Interval",
            "Robust Normalized Latency Coefficient + Error Confidence Interval",
        }
        assert len(indexer_rankings) > 0

    def test_perform_latency_linear_regression_with_empty_df(self):
        # Create an empty DataFrame
        empty_df = pd.DataFrame(
            columns=[
                "indexer",
                "deployment_hash",
                "indexer_network",
                "response_time_ms",
                "fee",
                "distance_miles",
                "score",
            ]
        )

        # Setup linear regression variables
        predictor = ["response_time_ms"]
        categorical = ["indexer", "deployment_hash", "indexer_network"]
        numeric = ["distance_miles", "fee"]

        # Check if the function raises an appropriate exception for empty DataFrame
        with pytest.raises(ValueError):
            perform_latency_linear_regression(empty_df, predictor, categorical, numeric)

    def test_perform_latency_linear_regression_with_missing_columns(self, sample_df):
        # Create a DataFrame with missing columns
        df_missing_columns = sample_df.drop(columns=["indexer", "fee"])

        # Setup linear regression variables
        predictor = ["response_time_ms"]
        categorical = ["indexer", "deployment_hash", "indexer_network"]
        numeric = ["distance_miles", "fee"]

        # Check if the function raises an appropriate exception for missing columns
        with pytest.raises(KeyError):
            perform_latency_linear_regression(
                df_missing_columns, predictor, categorical, numeric
            )

    def test_perform_latency_linear_regression_deterministic_verification(
        self, sample_df
    ):
        # Setup linear regression variables
        predictor = ["response_time_ms"]
        categorical = ["indexer", "deployment_hash", "indexer_network"]
        numeric = ["distance_miles", "fee"]

        # Perform linear regression twice and compare results
        (
            latency_linear_regression_indexer_rankings1,
            latency_linear_regression_results_df1,
        ) = perform_latency_linear_regression(
            sample_df, predictor, categorical, numeric
        )
        (
            latency_linear_regression_indexer_rankings2,
            latency_linear_regression_results_df2,
        ) = perform_latency_linear_regression(
            sample_df, predictor, categorical, numeric
        )

        # Check if the results are consistent across multiple runs
        pd.testing.assert_frame_equal(
            latency_linear_regression_indexer_rankings1,
            latency_linear_regression_indexer_rankings2,
        )

        # Check if the results are consistent across multiple runs
        pd.testing.assert_frame_equal(
            latency_linear_regression_results_df1, latency_linear_regression_results_df2
        )

    def test_perform_latency_linear_regression_original_df_unchanged(self, sample_df):
        # Create a copy of the original DataFrame
        original_df = sample_df.copy()

        # Setup linear regression variables
        predictor = ["response_time_ms"]
        categorical = ["indexer", "deployment_hash", "indexer_network"]
        numeric = ["distance_miles", "fee"]

        # Perform linear regression
        (_, _) = perform_latency_linear_regression(
            sample_df, predictor, categorical, numeric
        )

        # Check the original DataFrame is unchanged
        pd.testing.assert_frame_equal(original_df, sample_df)


class TestCalculateIndexerSuccessRate:
    @pytest.fixture
    def sample_df(self):
        return pd.DataFrame(
            {
                "indexer": [
                    "0xABC",
                    "0xXYZ",
                    "0x123",
                    "0xABC",
                    "0xXYZ",
                    "0x123",
                    "0xABC",
                    "0xXYZ",
                    "0x123",
                ],
                "status": [
                    "200 OK",
                    "404 Not Found",
                    "Unavailable(MissingBlock)",
                    "500 Internal Server Error",
                    "200 OK",
                    "200 OK",
                    "Unavailable(MissingBlock)",
                    "200 OK",
                    "403 Forbidden",
                ],
            }
        )

    def test_calculate_indexer_success_rate_basic(self, sample_df):
        result = calculate_indexer_success_rate(sample_df)

        # Check the structure of the result
        assert isinstance(result, pd.DataFrame)
        assert set(result.columns) == {"indexer", "average_status"}

        # Check the content of the result
        expected_result = pd.DataFrame(
            {
                "indexer": ["0x123", "0xABC", "0xXYZ"],
                "average_status": [2 / 3, 2 / 3, 2 / 3],
            }
        )
        pd.testing.assert_frame_equal(result, expected_result, check_exact=False)

    def test_calculate_indexer_success_rate_all_fail(self):
        df = pd.DataFrame(
            {
                "indexer": ["0xABC", "0xXYZ", "0x123"],
                "status": [
                    "404 Not Found",
                    "500 Internal Server Error",
                    "403 Forbidden",
                ],
            }
        )
        result = calculate_indexer_success_rate(df)
        assert all(result["average_status"] == 0.0)

    def test_calculate_indexer_success_rate_empty_df(self):
        df = pd.DataFrame(columns=["indexer", "status"])
        result = calculate_indexer_success_rate(df)
        assert result.empty

    def test_calculate_indexer_success_rate_case_sensitivity(self):
        df = pd.DataFrame(
            {
                "indexer": ["0xABC", "0xABC", "0xABC"],
                "status": ["200 OK", "200 ok", "200 Ok"],
            }
        )
        result = calculate_indexer_success_rate(df)
        assert result.loc[0, "average_status"] == 1 / 3

    def test_calculate_indexer_success_rate_sorting(self):
        df = pd.DataFrame(
            {
                "indexer": ["0xABC", "0xXYZ", "0x123", "0x789"],
                "status": [
                    "200 OK",
                    "200 OK",
                    "404 Not Found",
                    "Unavailable(MissingBlock)",
                ],
            }
        )
        result = calculate_indexer_success_rate(df)
        assert list(result["indexer"]) == ["0x123", "0x789", "0xABC", "0xXYZ"]

    def test_calculate_indexer_success_rate_large_dataset(self):
        np.random.seed(42)
        large_df = pd.DataFrame(
            {
                "indexer": np.random.choice(
                    ["0xABC", "0xXYZ", "0x123", "0x789", "0x456"], 100_000
                ),
                "status": np.random.choice(
                    [
                        "200 OK",
                        "Unavailable(MissingBlock)",
                        "404 Not Found",
                        "500 Internal Server Error",
                    ],
                    100_000,
                ),
            }
        )
        result = calculate_indexer_success_rate(large_df)
        assert len(result) == 5
        assert all(0 <= rate <= 1 for rate in result["average_status"])

    def test_calculate_indexer_success_rate_original_df_unchanged(self, sample_df):
        original_df = sample_df.copy()
        _ = calculate_indexer_success_rate(sample_df)
        pd.testing.assert_frame_equal(original_df, sample_df)


class TestCalculateIndexerUptime:
    @pytest.fixture
    def sample_df(self):
        return pd.DataFrame(
            {
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
                    "200 OK",
                    "200 OK",
                    "Error",
                    "200 OK",
                    "200 OK",
                    "Unavailable(MissingBlock)",
                    "200 OK",
                ],
            }
        )

    def test_calculate_indexer_uptime_base_case(self, sample_df):
        # Test the basic functionality of the function
        result = calculate_indexer_uptime(sample_df)

        # Check if the result has the expected columns
        expected_columns = [
            "indexer",
            "observed_duration_restricted",
            "uptime_duration_restricted",
            "observed_duration_full",
            "uptime_duration_full",
            "% up_y",
            "% up_x",
        ]
        result_columns = set(result.columns)
        expected_columns_set = set(expected_columns)

        missing_columns_not_in_result = expected_columns_set - result_columns
        unexpected_columns_in_result = result_columns - expected_columns_set

        assert not missing_columns_not_in_result and not unexpected_columns_in_result

        # Check if all indexers are present in the result
        assert set(result["indexer"]) == set(sample_df["indexer"])

        # Check that all percentages are either between 0 and 100, or nan's (where there was only 1 query)
        assert all(
            (0 <= percent <= 100) or np.isnan(percent) for percent in result["% up_x"]
        )
        assert all(
            (0 <= percent <= 100) or np.isnan(percent) for percent in result["% up_y"]
        )

    def test_calculate_indexer_uptime_all_up(self):
        # Test with all indexers being up
        df = pd.DataFrame(
            {
                "indexer": ["A", "A", "B", "B"],
                "timestamp": [
                    datetime(2024, 1, 1, 12, 0),
                    datetime(2024, 1, 1, 12, 2),
                    datetime(2024, 1, 1, 12, 0),
                    datetime(2024, 1, 1, 12, 2),
                ],
                "status": [
                    "200 OK",
                    "Unavailable(MissingBlock)",
                    "200 OK",
                    "Unavailable(MissingBlock)",
                ],
            }
        )
        result = calculate_indexer_uptime(df)

        # Confirm all percentages are 100%
        assert all(result["% up_x"] == 100)
        assert all(result["% up_y"] == 100)

    def test_calculate_indexer_uptime_all_down(self):
        # Test with all indexers being down
        df = pd.DataFrame(
            {
                "indexer": ["A", "A", "B", "B"],
                "timestamp": [
                    datetime(2024, 1, 1, 12, 0),
                    datetime(2024, 1, 1, 12, 2),
                    datetime(2024, 1, 1, 12, 0),
                    datetime(2024, 1, 1, 12, 2),
                ],
                "status": ["Error", "Bad", "Bad 504", "Error"],
            }
        )
        result = calculate_indexer_uptime(df)

        # Confirm all percentages are 0%
        assert all(result["% up_x"] == 0)
        assert all(result["% up_y"] == 0)

    def test_calculate_indexer_uptime_threshold(self):
        # Test the effect of the threshold parameter
        df = pd.DataFrame(
            {
                "indexer": ["A", "A", "A"],
                "timestamp": [
                    datetime(2024, 1, 1, 12, 0),
                    datetime(2024, 1, 1, 12, 5),
                    datetime(2024, 1, 1, 12, 10),
                ],
                "status": ["200 OK", "200 OK", "200 OK"],
            }
        )

        # Test with default threshold (120 seconds)
        result_default = calculate_indexer_uptime(df)

        # Test with a lower threshold (60 seconds)
        result_low_threshold = calculate_indexer_uptime(df, threshold_seconds=60)

        # The restricted uptime should be lower with the lower threshold
        assert (
            result_low_threshold["uptime_duration_restricted"].iloc[0]
            == result_default["uptime_duration_restricted"].iloc[0] / 2
        )

    def test_calculate_indexer_uptime_empty_df(self):
        # Test with an empty DataFrame
        df = pd.DataFrame(columns=["indexer", "timestamp", "status"])
        result = calculate_indexer_uptime(df)

        # The result should be an empty DataFrame with the expected columns
        assert result.empty
        expected_columns = [
            "indexer",
            "observed_duration_restricted",
            "uptime_duration_restricted",
            "% up_x",
            "observed_duration_full",
            "uptime_duration_full",
            "% up_y",
        ]
        assert all(col in result.columns for col in expected_columns)

    def test_calculate_indexer_uptime_single_entry_for_indexers(self):
        # Test with a DataFrame containing only one entry
        df = pd.DataFrame(
            {
                "indexer": ["A", "B"],
                "timestamp": [
                    datetime(2023, 1, 1, 12, 0),
                    datetime(2023, 1, 1, 12, 0),
                ],
                "status": ["200 OK", "BAD"],
            }
        )
        result = calculate_indexer_uptime(df)

        # Check if the result contains two rows
        assert len(result) == 2

        # All uptime's should be nan's
        assert np.isnan(result["% up_x"].iloc[0])
        assert np.isnan(result["% up_y"].iloc[0])
        assert np.isnan(result["% up_x"].iloc[1])
        assert np.isnan(result["% up_y"].iloc[1])

    def test_calculate_indexer_uptime_sorting(self):
        # Test if the result is sorted by '% up' in descending order
        df = pd.DataFrame(
            {
                "indexer": ["A", "A", "B", "B", "C", "C"],
                "timestamp": [datetime(2024, 1, 1, 12, i) for i in range(6)],
                "status": ["200 OK", "Error", "200 OK", "200 OK", "200 OK", "200 OK"],
            }
        )
        result = calculate_indexer_uptime(df)

        # Check if the '% up' column is sorted in descending order
        assert list(result["% up_x"]) == sorted(result["% up_x"], reverse=True)

    def test_calculate_indexer_uptime_rounding(self):
        # Test if the percentages are rounded to 3 decimal places
        df = pd.DataFrame(
            {
                "indexer": ["A", "A", "A"],
                "timestamp": [
                    datetime(2024, 1, 1, 12, 0),
                    datetime(2024, 1, 1, 12, 1),
                    datetime(2024, 1, 1, 12, 3),
                ],
                "status": ["200 OK", "Error", "200 OK"],
            }
        )
        result = calculate_indexer_uptime(df)

        # Check if the percentages are rounded to 3 decimal places
        assert all(round(percent, 3) == percent for percent in result["% up_x"])
        assert all(round(percent, 3) == percent for percent in result["% up_y"])


class TestCalculateStakeToFees:
    @pytest.fixture
    def sample_stake_query_pandas(self):
        return pd.DataFrame(
            {
                "indexer": ["A", "B", "C", "D", "E"],
                "stake_to_fees": [1.0, 2.0, 3.0, 4.0, 5.0],
                "other_column": [10, 20, 30, 40, 50],
            }
        )

    def test_calculate_indexer_stake_to_fees_base_case(self, sample_stake_query_pandas):
        # Calculate result
        result = calculate_indexer_stake_to_fees(sample_stake_query_pandas)

        # Check the result has the correct columns
        assert set(result.columns) == {
            "indexer",
            "stake_to_fees",
            "stake_to_fees_iqr_deviation",
        }

        # Check that 'stake_to_fees' column is unchanged
        pd.testing.assert_series_equal(
            result["stake_to_fees"], sample_stake_query_pandas["stake_to_fees"]
        )

        # Check that 'stake_to_fees_iqr_deviation' is calculated correctly
        median = 3.0
        q1 = 2.0
        q3 = 4.0
        iqr = q3 - q1
        expected_deviations = (
            sample_stake_query_pandas["stake_to_fees"] - median
        ) / iqr
        pd.testing.assert_series_equal(
            result["stake_to_fees_iqr_deviation"],
            expected_deviations,
            check_names=False,
        )

    def test_calculate_indexer_stake_to_fees_empty_df(self):
        # Create empty df
        empty_df = pd.DataFrame(columns=["indexer", "stake_to_fees"])

        # Calculate result
        result = calculate_indexer_stake_to_fees(empty_df)

        assert result.empty
        assert set(result.columns) == {
            "indexer",
            "stake_to_fees",
            "stake_to_fees_iqr_deviation",
        }

    def test_calculate_indexer_stake_to_fees_single_row(self):
        single_row_df = pd.DataFrame({"indexer": ["A"], "stake_to_fees": [1.0]})
        result = calculate_indexer_stake_to_fees(single_row_df)

        # Result should be nan because IQR in this case is 0 and /0 is nan.
        assert len(result) == 1
        assert pd.isna(result["stake_to_fees_iqr_deviation"].iloc[0])

    def test_calculate_indexer_stake_to_fees_with_nan_values(self):
        df_with_nan = pd.DataFrame(
            {
                "indexer": ["A", "B", "C", "D", "E"],
                "stake_to_fees": [1.0, np.nan, 3.0, np.nan, 5.0],
            }
        )
        result = calculate_indexer_stake_to_fees(df_with_nan)

        # Check that NaN values are handled correctly
        assert result["stake_to_fees_iqr_deviation"].isna().sum() == 2

    def test_calculate_indexer_stake_to_fees_constant_values(self):
        constant_df = pd.DataFrame(
            {
                "indexer": ["A", "B", "C", "D", "E"],
                "stake_to_fees": [3.0, 3.0, 3.0, 3.0, 3.0],
            }
        )
        result = calculate_indexer_stake_to_fees(constant_df)

        # All deviations should be NaN when all values are the same (IQR = 0)
        assert result["stake_to_fees_iqr_deviation"].isna().all()

    def test_calculate_indexer_stake_to_fees_extreme_values(self):
        extreme_df = pd.DataFrame(
            {"indexer": ["A", "B", "C"], "stake_to_fees": [1e9, 1e-9, 1e18]}
        )
        result = calculate_indexer_stake_to_fees(extreme_df)

        # Check that the function doesn't crash with extreme values
        assert len(result) == 3
        assert not result["stake_to_fees_iqr_deviation"].isna().any()

    def test_calculate_indexer_stake_to_fees_preserves_input(
        self, sample_stake_query_pandas
    ):
        original = sample_stake_query_pandas.copy()
        calculate_indexer_stake_to_fees(sample_stake_query_pandas)

        # Check that the input DataFrame is unchanged
        pd.testing.assert_frame_equal(sample_stake_query_pandas, original)


class TestAggregateIndexerInfo:
    def test_aggregate_indexer_info_base_case(self):
        ## Given
        sample_df = pd.DataFrame(
            {
                "indexer": ["A", "A", "B", "B", "C", "C", "C"],
                "org": ["X", "X", "Y", "Z", "W", "W", "W"],
                "dst_lat": [10.1, 13.123445, 35, 31, 55, 45, 50],
                "dst_lon": [22, 25.123445, 44, 41, 65, 60, 60],
            }
        )

        ## When
        result = aggregate_indexer_info(sample_df)

        ## Then
        assert list(result["org"]) == ["X", "Y", "W"]
        assert list(result["dst_lat"]) == [20, 40, 40]
        assert list(result["dst_lon"]) == [20, 40, 60]
        assert list(result["indexer"]) == ["A", "B", "C"]

    def test_aggregate_indexer_info_empty_df(self):
        ## Given
        df = pd.DataFrame(columns=["indexer", "org", "dst_lat", "dst_lon"])

        ## When
        result = aggregate_indexer_info(df)

        ## Then
        assert result.empty
        assert list(result.columns) == ["indexer", "org", "dst_lat", "dst_lon"]

    def test_aggregate_indexer_info_with_nans(self):
        ## Given
        df = pd.DataFrame(
            {
                "indexer": ["A", "A", "B", "B", "B"],
                "org": [np.nan, "X", "Y", np.nan, np.nan],
                "dst_lat": [10, np.nan, np.nan, np.nan, np.nan],
                "dst_lon": [20, np.nan, np.nan, np.nan, np.nan],
            }
        )

        ## When
        result = aggregate_indexer_info(df)

        ## Then
        assert list(result["indexer"]) == ["A", "B"]
        assert list(result["org"]) == ["X", "Y"]

        dst_lat = list(result["dst_lat"])
        dst_lon = list(result["dst_lon"])
        assert dst_lat[0] == 0
        assert dst_lon[0] == 20
        assert np.isnan(dst_lat[1])
        assert np.isnan(dst_lon[1])


class TestMergeAndPrepareDataframes:
    @pytest.fixture
    def indexer_uptime(self):
        return pd.DataFrame(
            {
                "indexer": ["0xABC", "0xXYZ", "0x123"],
                "uptime": [99.5, 98.7, 97.0],
                "observed_duration_full": [100, 200, 300],
                "uptime_duration_full": [99, 197, 291],
            }
        )

    @pytest.fixture
    def indexer_rankings(self):
        return pd.DataFrame(
            {
                "indexer": ["0xABC", "0xXYZ", "0x789"],
                "ranking": [1, 2, 4],
                "% up_y": [95, 96, 97],
            }
        )

    @pytest.fixture
    def agg_df(self):
        return pd.DataFrame(
            {
                "indexer": ["0xABC", "0xXYZ", "0x456"],
                "Coefficient": [0.5, 0.3, np.nan],
                "Standard Error": [0.05, 0.03, 0.01],
                "p-value": [0.01, 0.02, np.nan],
            }
        )

    @pytest.fixture
    def indexer_success_rate(self):
        return pd.DataFrame(
            {"indexer": ["0xABC", "0xXYZ", "0xDEF"], "success_rate": [90, 85, 80]}
        )

    @pytest.fixture
    def stake_to_fees(self):
        return pd.DataFrame(
            {
                "indexer": ["0xABC", "0xXYZ", "0xGHI"],
                "stake_fees_ratio": [100, 200, 300],
            }
        )

    def test_merge_base_case(
        self,
        indexer_uptime,
        indexer_rankings,
        agg_df,
        indexer_success_rate,
        stake_to_fees,
    ):
        # Compute result
        result = merge_and_prepare_dataframes(
            indexer_uptime,
            indexer_rankings,
            agg_df,
            indexer_success_rate,
            stake_to_fees,
        )

        # Test that all indexers are present
        assert set(result["indexer"]) == {"0x123", "0xABC", "0xXYZ"}

        # Ensure existing_dips_agreements column is as expected
        assert "existing_dips_agreements" in result.columns
        assert all(result["existing_dips_agreements"] == 0)

        # Ensure avg_sync_duration column is as expected
        assert "avg_sync_duration" in result.columns
        assert all(pd.isna(result["avg_sync_duration"]))

        # Ensure indexing_agreement_acceptance_latency column is as expected
        assert "indexing_agreement_acceptance_latency" in result.columns
        assert all(pd.isna(result["indexing_agreement_acceptance_latency"]))

        # Columns correctly dropped
        assert "% up_y" not in result.columns
        assert "observed_duration_full" not in result.columns
        assert "uptime_duration_full" not in result.columns

    def test_merge__missing_indexer(
        self,
        indexer_uptime,
        indexer_rankings,
        agg_df,
        indexer_success_rate,
        stake_to_fees,
    ):
        # Remove an indexer from one DataFrame to simulate missing data
        indexer_uptime.drop(
            indexer_uptime.index[indexer_uptime["indexer"] == "0xABC"], inplace=True
        )
        result = merge_and_prepare_dataframes(
            indexer_uptime,
            indexer_rankings,
            agg_df,
            indexer_success_rate,
            stake_to_fees,
        )
        # '0xABC' should not be present as it was removed from `indexer_uptime`
        assert "0xABC" not in result["indexer"].values

    def test_merge_no_common_indexers(
        self,
        indexer_uptime,
        indexer_rankings,
        agg_df,
        indexer_success_rate,
        stake_to_fees,
    ):
        # Create a completely new set of indexers across the dataframes
        indexer_uptime["indexer"] = ["0xAAA", "0xBBB", "0xCCC"]

        # Compute the result
        result = merge_and_prepare_dataframes(
            indexer_uptime,
            indexer_rankings,
            agg_df,
            indexer_success_rate,
            stake_to_fees,
        )

        # Check that the result is not empty
        assert not result.empty

        # Check that all rows from indexer_uptime are present
        assert len(result) == len(indexer_uptime)
        assert set(result["indexer"]) == set(indexer_uptime["indexer"])

        # Check that columns from other DataFrames are present but contain only NaN values
        for col in [
            "ranking",
            "Coefficient",
            "Standard Error",
            "p-value",
            "success_rate",
            "stake_fees_ratio",
        ]:
            assert col in result.columns
            assert result[col].isna().all()

    def test_merge_additional_columns(
        self,
        indexer_uptime,
        indexer_rankings,
        agg_df,
        indexer_success_rate,
        stake_to_fees,
    ):
        # Add new columns to multiple input dataframes
        indexer_uptime["new_col_1"] = np.random.randn(len(indexer_uptime))
        indexer_rankings["new_col_2"] = np.random.randn(len(indexer_rankings))
        agg_df["new_col_3"] = np.random.randn(len(agg_df))

        # Compute result
        result = merge_and_prepare_dataframes(
            indexer_uptime,
            indexer_rankings,
            agg_df,
            indexer_success_rate,
            stake_to_fees,
        )

        # Check that all expected columns are present
        expected_columns = {
            "indexer",
            "uptime",
            "ranking",
            "Coefficient",
            "Standard Error",
            "p-value",
            "success_rate",
            "stake_fees_ratio",
            "existing_dips_agreements",
            "avg_sync_duration",
            "indexing_agreement_acceptance_latency",
        }
        assert all(col in result.columns for col in expected_columns)

        # Check that new columns are present too
        new_expected_columns = {"new_col_1", "new_col_2", "new_col_3"}
        assert all(col in result.columns for col in new_expected_columns)
