"""
Test suite covering the BigQueryProvider class and its associated functions.
"""

import socket
from datetime import datetime

import pandas as pd
import pytest

from iisa.bq import (
    BigQueryProvider,
    InitialQueryDataFrame,
    _get_combined_query,
    _get_initial_query,
    _get_initial_stake_to_fees_query,
)
from iisa.time import DateStr, TimestampStr


class TestGetCombinedQuery:
    """
    Tests for the query string generation functions.
    """

    def test_basic_query(self):
        # Given a start date, a number of days and a number of rows to use
        start_date = DateStr("2024-01-01")

        # When _get_combined_query is called
        query = _get_combined_query(start_date, 10, 20000000)

        # Then the query string should contain the expected date ranges
        assert (
            "WHERE day BETWEEN '2024-01-01' AND DATE_ADD('2024-01-01', INTERVAL 10 DAY"
            in query
        )
        assert (
            "WHERE day_partition BETWEEN '2024-01-01' AND DATE_ADD('2024-01-01', INTERVAL 10 DAY)"
            in query
        )

    def test_get_initial_query(self):
        # Given a start date and a number of days
        start_date = DateStr("2024-01-01")

        # When get_initial_query is called
        query = _get_initial_query(start_date, 10)

        # Then the query string should contain the expected date range
        assert (
            "BETWEEN '2024-01-01' AND DATE_ADD('2024-01-01', INTERVAL 10 DAY)" in query
        )

    def test_get_initial_stake_to_fees_query(self):
        # Given a start timestamp
        start_ts = TimestampStr("2024-01-01T00:00:00Z")

        # When _get_initial_stake_to_fees_query is called
        query = _get_initial_stake_to_fees_query(start_ts)

        # Then the query string should contain the expected timestamp where clause
        assert "WHERE TIMESTAMP(mia.day_partition) > '2024-01-01T00:00:00Z'" in query


class TestFetchData:
    """
    Tests for the fetch_initial_query_results function.

    This suite tests various scenarios for the fetch_initial_query_results function,
    including successful fetch, empty results, error handling, and retry mechanism.
    """

    @pytest.fixture
    def bigquery(self):
        return BigQueryProvider("graph-mainnet", "US")

    def test_successful_fetch(self, faker, mocker, bigquery):
        ## Given
        # Test timeframe
        start_date = datetime.strptime("2024-08-01", "%Y-%m-%d")
        num_days = 28

        # Setup sample data and the DataFrame to be returned by the 'to_pandas' method
        expected_df = InitialQueryDataFrame(
            {
                "deployment_hash": [faker.deployment_id() for _ in range(5)],
                "indexer": [faker.indexer_id() for _ in range(5)],
                "num_rows": [10, 20, 15, 5, 25],
                "timestamp": [
                    "2024-08-01T12:00:00Z",
                    "2024-08-01T13:00:00Z",
                    "2024-08-01T14:00:00Z",
                    "2024-08-01T15:00:00Z",
                    "2024-08-01T16:00:00Z",
                ],
                "status": ["success", "success", "failure", "success", "failure"],
            }
        )
        expected_df.sort_values(by="num_rows", ascending=False, inplace=True)

        # Mock object that read_gbq will return
        mock_query_job = mocker.Mock()
        mock_query_job.to_pandas.return_value = expected_df

        # Apply the mock to make read_gbq return the mock_query_job
        mocker.patch("bigframes.pandas.read_gbq", return_value=mock_query_job)

        ## When
        result_df = bigquery.fetch_initial_query_results(start_date, num_days)

        ## Then
        # Verify the result DataFrame is sorted correctly by 'num_rows'
        pd.testing.assert_frame_equal(result_df, expected_df)

        # Additional assert to explicitly check the order of 'num_rows' to ensure sorting is as expected
        assert (result_df["num_rows"].values == expected_df["num_rows"].values).all()

    def test_fetch_empty_data(self, mocker, bigquery):
        ## Given
        # Test timeframe
        start_date = datetime.strptime("2024-08-01", "%Y-%m-%d")
        num_days = 28

        # Setup sample data and the DataFrame to be returned by the 'to_pandas' method
        expected_df = InitialQueryDataFrame(
            {
                "deployment_hash": pd.Series(dtype="string"),
                "indexer": pd.Series(dtype="string"),
                "num_rows": pd.Series(dtype="int64"),
                "timestamp": pd.Series(dtype="string"),
                "status": pd.Series(dtype="string"),
            }
        )

        # Mock object that read_gbq will return
        mock_query_job = mocker.Mock()
        mock_query_job.to_pandas.return_value = expected_df

        # Apply the mock to make read_gbq return the mock_query_job
        mocker.patch("bigframes.pandas.read_gbq", return_value=mock_query_job)

        ## When
        result_df = bigquery.fetch_initial_query_results(start_date, num_days)

        ## Then
        # Assertions to check the result is an empty DataFrame
        assert result_df.empty

    def test_fail_on_generic_error(self, faker, mocker, bigquery):
        """
        Check the retry mechanism does not capture generic errors.
        """
        ## Given
        # Test timeframe
        start_date = datetime.strptime("2024-08-01", "%Y-%m-%d")
        num_days = 28

        # Setup sample data and the DataFrame to be returned by the 'to_pandas' method
        expected_df = InitialQueryDataFrame(
            {
                "deployment_hash": [faker.deployment_id() for _ in range(5)],
                "indexer": [faker.indexer_id() for _ in range(5)],
                "num_rows": [10, 20, 15, 5, 25],
                "timestamp": [
                    "2024-08-01T12:00:00Z",
                    "2024-08-01T13:00:00Z",
                    "2024-08-01T14:00:00Z",
                    "2024-08-01T15:00:00Z",
                    "2024-08-01T16:00:00Z",
                ],
                "status": ["success", "success", "failure", "success", "failure"],
            }
        )

        # Mock object that read_gbq will return
        mock_query_job = mocker.Mock()
        mock_query_job.to_pandas.return_value = expected_df

        # Apply the mock to make read_gbq return the mock_query_job, then apply a side effect.
        mock_read_gbq = mocker.patch(
            "bigframes.pandas.read_gbq", return_value=mock_query_job
        )
        mock_read_gbq.side_effect = Exception("Generic error. Query failed.")

        ## When
        # Call the function and assert that it raises an exception "Generic error. Query failed."
        with pytest.raises(Exception, match="Generic error. Query failed."):
            bigquery.fetch_initial_query_results(start_date, num_days)

    def test_rerty_on_connection_error(self, faker, mocker, bigquery):
        """
        Check the retry mechanism works as expected when a connection error is raised.
        """
        ## Given
        # Test timeframe
        start_date = datetime.strptime("2024-08-01", "%Y-%m-%d")
        num_days = 28

        # Setup sample data and the DataFrame to be returned by the 'to_pandas' method
        expected_df = pd.DataFrame(
            {
                "deployment_hash": [faker.deployment_id() for _ in range(5)],
                "indexer": [faker.indexer_id() for _ in range(5)],
                "num_rows": [10, 20, 15, 5, 25],
                "timestamp": [
                    "2024-08-01T12:00:00Z",
                    "2024-08-01T13:00:00Z",
                    "2024-08-01T14:00:00Z",
                    "2024-08-01T15:00:00Z",
                    "2024-08-01T16:00:00Z",
                ],
                "status": ["success", "success", "failure", "success", "failure"],
            }
        )
        expected_df.sort_values(by="num_rows", ascending=False, inplace=True)

        # Create a Mock object for the to_pandas method to simulate connection error on first call
        mock_query_job = mocker.Mock()
        mock_query_job.to_pandas.side_effect = [
            ConnectionError(
                "Temporary connectivity issue"
            ),  # First call raises an error
            socket.timeout(
                "Connection timed out"
            ),  # Second call raises a different error
            expected_df,  # Second call returns the DataFrame
        ]

        # Apply the mock to make read_gbq return the mock_query_job
        mocker.patch("bigframes.pandas.read_gbq", return_value=mock_query_job)

        ## When
        # Call the fetch_initial_query_results function, which should retry after the first connection error
        result_df = bigquery.fetch_initial_query_results(start_date, num_days)

        ## Then
        # Assert that the result DataFrame is sorted correctly by 'num_rows'
        assert not result_df.empty
        assert result_df.equals(expected_df)
