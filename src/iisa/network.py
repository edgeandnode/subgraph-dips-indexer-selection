"""
A module that provides classes to represent the graph network of indexers and their indexed subgraphs.

.. note::
    The classes in this module are meant to be used as data transfer objects (DTOs) to represent the graph network
    of indexers and their indexed subgraphs. The classes are not meant to be used as domain objects.
"""

from typing import Iterable, Optional, cast

import pandera as pa
from pandera.typing import DataFrame, Series

from .geoip import GeoipResolver
from .typing import (
    EthAddressField,
    HttpUrlField,
    HttpUrlStr,
    IndexerId,
    IpV4AddressField,
    Iso3166CountryField,
    LatitudeField,
    LongitudeField,
)


class Indexer:
    """
    An indexer.

    Represents an indexer with a unique indexer ID and a URL.
    """

    def __init__(
        self,
        indexer_id: IndexerId,
        url: HttpUrlStr,
    ) -> None:
        """
        Initializes a new instance of the Indexer class.

        :param indexer_id: The unique indexer ID.
        :param url: The URL of the indexer.
        """
        self._id = indexer_id
        self._url = url

    @property
    def id(self) -> IndexerId:
        """
        The unique indexer ID.

        :returns: The indexer ID.
        """
        return self._id

    @property
    def url(self) -> HttpUrlStr:
        """
        The URL of the indexer.

        :returns: The URL of the indexer.
        """
        return self._url


class IndexersSchema(pa.DataFrameModel):
    """A schema for validating the indexers dataframe"""

    indexer: Series[str] = EthAddressField(
        description="The indexer's Ethereum address", unique=True
    )
    url: Series[str] = HttpUrlField(description="The indexer's URL")
    indexer_network: Series[str] = pa.Field(isin=["arbitrum"])

    # Resolved IP address and geolocation information
    ip_addr: Series[str] = IpV4AddressField(
        description="The indexer's IP address", nullable=True
    )
    org: Series[str] = pa.Field(
        description="The organization name of the indexer's IP address", nullable=True
    )
    country: Series[str] = Iso3166CountryField(
        description="The country code of the indexer's IP address geolocation",
        nullable=True,
    )
    latitude: Series[float] = LatitudeField(
        description="The latitude (decimal) of the indexer's IP address geolocation",
        nullable=True,
    )
    longitude: Series[float] = LongitudeField(
        description="The longitude (decimal) of the indexer's IP address geolocation",
        nullable=True,
    )


IndexersDataFrame = DataFrame[IndexersSchema]


class NetworkProvider:
    """
    The Graph network information provider.

    The network provider is responsible for holding the network information snapshot, which includes the list of
    indexers and their indexed subgraphs. If the network information snapshot is not available, the provider will raise
    a ValueError when attempting to access the network information.
    """

    def __init__(self, geoip: GeoipResolver) -> None:
        """Initializes a new instance of the NetworkProvider class."""
        self._geoip = geoip

        self._indexers: Optional[IndexersDataFrame] = None

    def set_snapshot(
        self,
        indexers: Iterable[Indexer],
    ) -> None:
        """
        Updates the network snapshot with the provided indexers and subgraphs.

        .. note::
            This method is meant to be called by the Rust host to update the network snapshot with the latest
            network subgraph retrieval results.

        :param indexers: The list of indexers.
        """
        self._indexers = _to_indexers_dataframe(indexers, geoip=self._geoip)

    def indexers(self) -> IndexersDataFrame:
        """
        The list of indexers.

        :returns: The indexers dataframe.
        """
        if self._indexers is None:
            raise ValueError("Network snapshot not available")

        return self._indexers


def _to_indexers_dataframe(
    indexers: Iterable[Indexer],
    *,
    geoip: GeoipResolver,
) -> IndexersDataFrame:
    """
    Converts a list of indexers to a Pandas DataFrame.

    :param indexers: The list of indexers.
    :param geoip: The GeoipResolver instance.
    :return: The indexers DataFrame.
    """
    indexers_df: DataFrame = DataFrame(
        {
            "indexer": [indexer.id for indexer in indexers],
            "url": [indexer.url for indexer in indexers],
        }
    )

    # Set network columns to 'arbitrum'
    indexers_df["indexer_network"] = "arbitrum"

    # Resolve the IP address and geolocation information for each indexer
    indexers_df[["ip_addr", "org", "country", "latitude", "longitude"]] = indexers_df[
        "url"
    ].apply(lambda url: Series(geoip.resolve_url_host_info(url)))

    return cast(IndexersDataFrame, indexers_df)
