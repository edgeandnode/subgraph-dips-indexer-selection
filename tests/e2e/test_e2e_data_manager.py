"""
Integration tests covering E2E the DataManager class.
"""

import logging
import os

import pytest

from iisa import BigQueryProvider, DataManager


@pytest.fixture(scope="module")
def bigquery_provider():
    """
    A bigquery provider fixture.
    """
    return BigQueryProvider("graph-mainnet", "US")


@pytest.mark.skipif(
    "CI" in os.environ,
    reason="Skip test in CI: Requires access to Google BigQuery",
)
def test_load_scores(bigquery_provider):
    """Test that DataManager can load pre-computed scores from BigQuery."""
    logging.basicConfig(level=logging.DEBUG, handlers=[logging.StreamHandler()])

    ## Given
    data_manager = DataManager(bigquery_provider)

    ## When
    success = data_manager.load_scores()

    ## Then
    assert success, "load_scores() should return True when scores exist"

    processed_data = data_manager.get_data()
    assert processed_data is not None
    assert not processed_data.empty
    assert processed_data.shape[0] > 0

    # Verify expected columns exist
    assert "indexer" in processed_data.columns
    assert "url" in processed_data.columns

    # Verify scores age is reasonable (less than 7 days)
    scores_age = data_manager.get_scores_age()
    assert scores_age is not None
    assert scores_age.days < 7, f"Scores are {scores_age.days} days old"
