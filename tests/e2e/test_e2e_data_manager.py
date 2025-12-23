"""
Integration tests covering E2E the DataManager class.
"""

import logging
import os

import pytest

from __fixtures__.network import load_fixture_data
from iisa import BigQueryProvider, DataManager, GeoipResolver, NetworkProvider
from iisa.data_manager import DEFAULT_NUM_DAYS, DEFAULT_TARGET_ROWS


@pytest.fixture(scope="module")
def bigquery_provider():
    """
    A bigquery provider fixture.
    """
    return BigQueryProvider("graph-mainnet", "US")


@pytest.fixture(scope="module")
def geoip_resolver(ipinfo_io_auth):
    """
    A GeoIP resolver fixture.
    """
    return GeoipResolver(ipinfo_io_auth)


@pytest.fixture()
def network_provider(geoip_resolver):
    """
    A network provider fixture.
    """
    network_provider = NetworkProvider(geoip_resolver)

    # Load the network data fixture
    network_indexers_info = load_fixture_data()
    network_provider.set_snapshot(network_indexers_info)

    return network_provider


@pytest.mark.skipif(
    "CI" in os.environ,
    reason="Skip test in CI: Requires access to Google BigQuery",
)
def test_fetch_and_update(bigquery_provider, network_provider):
    logging.basicConfig(level=logging.DEBUG, handlers=[logging.StreamHandler()])

    ## Given
    data_manager = DataManager(bigquery_provider, network_provider)

    ## When
    data_manager.fetch_data_and_update(
        num_days=DEFAULT_NUM_DAYS,
        target_rows=DEFAULT_TARGET_ROWS,
    )

    processed_data = data_manager.get_data()
    indexer_rankings = data_manager.get_latency_linear_regression_indexer_rankings()
    regression_results = data_manager.get_latency_linear_regression_results()

    ## Then
    # Assert processed data is not empty
    assert processed_data is not None
    assert not processed_data.empty
    assert processed_data.shape[0] > 0

    # Assert indexer rankings is not empty
    assert indexer_rankings is not None
    assert not indexer_rankings.empty
    assert indexer_rankings.shape[0] > 0

    # Assert regression results is not empty
    assert regression_results is not None
    assert not regression_results.empty
    assert regression_results.shape[0] > 0
