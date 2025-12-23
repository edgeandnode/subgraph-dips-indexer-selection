"""
Test suite for the network module.
"""

import pytest

from __fixtures__ import network as network_fixture
from iisa.geoip import GeoipResolver
from iisa.network import IndexersSchema, NetworkProvider


@pytest.mark.skip(reason="requires a new IPInfo.io API key")
class TestNetworkIndexersDataframe:
    def test_indexers_dataframe_conversion(self, ipinfo_io_auth):
        ## Given
        resolver = GeoipResolver(ipinfo_io_auth)
        provider = NetworkProvider(geoip=resolver)

        # Initialize the network provider with test data
        test_data = network_fixture.load_fixture_data()
        provider.set_snapshot(test_data)

        ## When
        indexers = provider.indexers()

        ## Then
        # Validate the indexers with the dataframe schema
        IndexersSchema.validate(indexers)

        # Asert that the indexers dataframe is not empty
        assert not indexers.empty

        # Assert that at least one row has a non-null IP address and Geolocation info
        assert indexers["ip_addr"].notnull().any()
        assert indexers["org"].notnull().any()
        assert indexers["country"].notnull().any()
        assert indexers["latitude"].notnull().any()
        assert indexers["longitude"].notnull().any()
