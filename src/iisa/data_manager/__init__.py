"""
The DataManager is responsible for fetching, processing, and analyzing BigQuery data on a daily basis.
This class is instantiated once and reused as needed to ensure efficient data management throughout its lifecycle.

Responsibilities:
    - Fetches data from BigQuery using specified queries and parameters.
    - Processes the retrieved data by applying various transformations and calculations.
    - Performs statistical analysis and machine learning tasks such as linear regression.
    - Aggregates and merges additional information from multiple data sources.
    - Prepares the data for further use by other components or services.
"""

from .manager import (
    DEFAULT_NUM_DAYS,
    DEFAULT_TARGET_ROWS,
    DataManager,
    IndexerRankingsDataFrame,
    LinearRegressionResultsDataFrame,
)

__all__ = [
    "DataManager",
    "IndexerRankingsDataFrame",
    "LinearRegressionResultsDataFrame",
    "DEFAULT_NUM_DAYS",
    "DEFAULT_TARGET_ROWS",
]
