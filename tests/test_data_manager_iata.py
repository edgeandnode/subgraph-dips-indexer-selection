"""
Test suite covering the IATA code to geolocation info mapping.
"""

import pytest

from iisa.data_manager.iata import get_iata_geolocation_info


class TestIATAInfo:
    @pytest.fixture
    def iata_geolocation_info(self):
        return get_iata_geolocation_info()

    def test_iata_code_to_geolocation_info(self, iata_geolocation_info):
        ## Given

        iata_code = "LAX"

        ## When
        iata_info = iata_geolocation_info.loc[iata_code]

        ## Then
        assert iata_info["latitude"] == 33.9425
        assert iata_info["longitude"] == -118.40805
        assert iata_info["country"] == "US"

    def test_invalid_iata_code_to_geolocation_info(self, iata_geolocation_info):
        ## Given
        iata_code = "XXX"

        ## When
        with pytest.raises(KeyError):
            _ = iata_geolocation_info.loc[iata_code]
