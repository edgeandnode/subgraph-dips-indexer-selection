import pandera as pa
from pandera.typing import DataFrame, Series

from .typing import DeploymentIdField, HttpUrlField, IndexerIdField, QueryIdField


class PerfHistorySchema(pa.DataFrameModel):
    """
    Schema for the performance history dataset.

    The curated performance history dataset returned by the data manager class.
    """

    query_id: Series[str] = QueryIdField()
    deployment_hash: Series[str] = DeploymentIdField()
    indexer: Series[str] = IndexerIdField()
    url: Series[str] = HttpUrlField()


PerfHistoryDataFrame = DataFrame[PerfHistorySchema]
