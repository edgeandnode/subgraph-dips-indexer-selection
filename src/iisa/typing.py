"""
IISA-specific type hints and Pandera schema field factories.
"""

from functools import partial
from typing import NewType, Type, TypeVar

import pandas as pd
import pandera as pa
import pyarrow
from pandera import DataFrameModel
from pandera.typing import DataFrame

BASE58_ALPHABET = "123456789ABCDEFGHJKLMNPQRSTUVWXYZabcdefghijkmnopqrstuvwxyz"

QueryIdStr = NewType("QueryIdStr", str)
HttpUrlStr = NewType("HttpUrlStr", str)
SubgraphId = NewType("SubgraphId", str)
IpfsHashStr = NewType("IpfsHashStr", str)
DeploymentId = IpfsHashStr
EthAddressStr = NewType("EthAddressStr", str)
IndexerId = EthAddressStr

IataCodeField = partial(
    pa.Field,
    description="IATA 3-letter Location code",
    str_matches=r"^[A-Z]{3}$",
)

LatitudeField = partial(
    pa.Field,
    description="Latitude in decimal degrees",
    in_range={"min_value": -90.0, "max_value": 90.0},
)

LongitudeField = partial(
    pa.Field,
    description="Longitude in decimal degrees",
    in_range={"min_value": -180.0, "max_value": 180.0},
)

Iso3166CountryField = partial(
    pa.Field,
    description="ISO 3166-1 alpha-2 country code",
    str_matches=r"^[A-Z]{2}$",
)

ArrowDate32Field = partial(
    pa.Field,
    description="Date in pyarrow.date32 format",
    dtype_kwargs={"pyarrow_dtype": pyarrow.date32()},
    coerce=True,
)

EthAddressField = partial(
    pa.Field, description="Ethereum address string", str_matches=r"0x[a-fA-F0-9]{40}"
)

IndexerIdField = partial(EthAddressField, description="Indexer ID string")

HttpUrlField = partial(
    pa.Field, description="HTTP URL string", str_matches=r"https?://[^\s]+"
)

IpV4AddressField = partial(
    pa.Field,
    description="IPv4 address string",
    str_matches=r"^\d{1,3}\.\d{1,3}\.\d{1,3}\.\d{1,3}$",
)

IpfsHashField = partial(
    pa.Field, description="IPFS hash string", str_matches=r"Qm[a-zA-Z0-9]{44}"
)

DeploymentIdField = partial(IpfsHashField, description="Deployment ID string")

SubgraphIdField = partial(
    pa.Field, description="Subgraph ID string", str_matches=f"[{BASE58_ALPHABET}]{{44}}"
)

QueryIdField = partial(
    pa.Field, description="Query ID string", str_matches=r"[a-f0-9]{16}-[A-Z]{3}"
)

_S = TypeVar("_S", bound=DataFrameModel)


def empty_dataframe(
    model: Type[_S],
) -> DataFrame[_S]:
    """
    Helper method to create an empty pandas dataframe from a pandera schema.

    See: https://stackoverflow.com/questions/76630592

    :param model: Pandera DataFrameModel.
    :return: An empty pandas DataFrame.
    """
    schema = model.to_schema()

    column_names = list(schema.columns.keys())
    column_types = {column: str(dtype) for column, dtype in schema.dtypes.items()}

    dataframe = pd.DataFrame(columns=column_names).astype(column_types)

    return DataFrame[_S](dataframe)
