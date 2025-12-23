from pathlib import Path
from typing import cast

import airportsdata
import pandas as pd
import pandera as pa
from pandera.typing import DataFrame, Index, Series

from ..typing import IataCodeField, Iso3166CountryField, LatitudeField, LongitudeField

__all__ = [
    "IataInfoDataFrame",
    "get_iata_geolocation_info",
]


class IataInfoSchema(pa.DataFrameModel):
    """
    Schema for the iata_geolocation DataFrame.
    """

    IATA_code: Index[str] = IataCodeField(unique=True)
    latitude: Series[float] = LatitudeField(
        description="Latitude (decimal) of the airport reference point"
    )
    longitude: Series[float] = LongitudeField(
        description="Longitude (decimal) of the airport reference point"
    )
    country: Series[str] = Iso3166CountryField()


IataInfoDataFrame = DataFrame[IataInfoSchema]


@pa.check_types
def _load_airportsdata_iata_pandas() -> IataInfoDataFrame:
    """
    Load the IATA airport data from the airportsdata package.

    Additionally, at load time, check the data frame content matches the IataInfoSchema.

    :returns: DataFrame containing IATA airport data.
    """
    airportsdata_csv = Path(airportsdata.__file__).parent / "airports.csv"

    iata_df = pd.read_csv(
        airportsdata_csv,
        usecols=["iata", "lat", "lon", "country"],
        na_values={"iata": [""], "country": [""]},
        keep_default_na=False,
    )
    iata_df.rename(
        columns={"iata": "IATA_code", "lat": "latitude", "lon": "longitude"},
        inplace=True,
    )
    iata_df.dropna(subset=["IATA_code"], inplace=True)
    iata_df.set_index("IATA_code", inplace=True)
    iata_df.sort_index(inplace=True)

    return cast(IataInfoDataFrame, iata_df)


# Load once, at import time, the IATA airport data from the airportsdata package
_airportsdata_iata = _load_airportsdata_iata_pandas()


def get_iata_geolocation_info() -> IataInfoDataFrame:
    """
    Get the IATA airport geolocation data.

    :returns: DataFrame containing IATA airport geolocation data.
    """
    return _airportsdata_iata
