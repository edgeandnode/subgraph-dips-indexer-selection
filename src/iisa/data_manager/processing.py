"""
Helper functions for the data_manager module.
"""

from typing import Optional, Tuple, overload

import numpy as np
import pandas as pd
import pandera as pa
from numpy.linalg import pinv
from pandera.typing import DataFrame, Series
from scipy.stats import t
from sklearn.compose import ColumnTransformer
from sklearn.linear_model import LinearRegression
from sklearn.metrics import mean_squared_error
from sklearn.pipeline import Pipeline
from sklearn.preprocessing import OneHotEncoder, StandardScaler

from .iata import IataInfoDataFrame, get_iata_geolocation_info
from ..bq import StakeToFeesDataFrame
from ..network import IndexersDataFrame
from ..typing import (
    HttpUrlField,
    IataCodeField,
    IndexerIdField,
    IpV4AddressField,
    Iso3166CountryField,
    LatitudeField,
    LongitudeField,
    QueryIdField,
)

# Constants
LATENCY_COEFFICIENT_STANDARD_ERROR_MULTIPLIER = 1.5

# Request status codes and error messages
REQUEST_STATUS_OK = "200 OK"
REQUEST_STATUS_UNAVAILABLE_MISSING_BLOCK = "Unavailable(MissingBlock)"


def adjust_rows(initial_query_results_pandas: pd.DataFrame, target_rows: int) -> int:
    """
    Dynamically adjust the number of rows per group to approximate a target total number of rows.

    This function iteratively adjusts the upper limit of rows for each group (defined by 'deployment_hash'
    and 'indexer') in the DataFrame to ensure that the sum of restricted rows is close to the specified
    target number of rows. It decreases or increases the upper limit based on the difference between the
    current sum and the target, and stops when the sum is within a specified tolerance or a maximum number
    of iterations is reached.

    :param initial_query_results_pandas: DataFrame containing the initial query results with a 'num_rows' column.
    :param target_rows: The target total number of rows for the DataFrame.
    :returns: The adjusted upper limit for the number of rows per group.
    """
    if target_rows < 0:
        raise ValueError("Target rows must be a non-negative integer")

    x = 1_000  # Starting estimate for the number of rows to record for each ['deployment_hash', 'indexer'] combination.
    initial_query_results_pandas["num_rows_restricted"] = initial_query_results_pandas[
        "num_rows"
    ].clip(upper=x)
    tolerance = target_rows * 0.01  # 1% tolerance range
    max_iterations = 1_000  # Maximum number of iterations to avoid infinite loops
    iteration = 0

    while not (
        target_rows - tolerance
        <= initial_query_results_pandas["num_rows_restricted"].sum()
        <= target_rows + tolerance
    ):
        current_sum = initial_query_results_pandas["num_rows_restricted"].sum()
        if current_sum > target_rows:
            x = int(x * 0.99)  # Decrease x by 1%
        elif current_sum < target_rows:
            x = int(x * 1.01)

        initial_query_results_pandas["num_rows_restricted"] = (
            initial_query_results_pandas["num_rows"].clip(upper=x)
        )
        iteration += 1

        # Break the loop if the difference between the current sum and the target is within the
        # tolerance range or if the maximum number of iterations is reached.
        if abs(current_sum - target_rows) <= tolerance or iteration >= max_iterations:
            break

    return initial_query_results_pandas["num_rows_restricted"].max()


class _MergeInIndexersInfoInputSchema(pa.DataFrameModel):
    """
    Schema for the merge_in_indexers_info function input DataFrame.
    """

    indexer: Series[str] = IndexerIdField()
    url: Series[str] = HttpUrlField()


class _MergeInIndexersInfoMixinSchema(pa.DataFrameModel):
    """
    Schema definition of the new columns to be added to the combined query data.

    This is an intermediate datatype result of merge_in_indexers_info function.
    """

    indexer_network: Series[str] = pa.Field(isin=["arbitrum"], nullable=True)
    ip_addr: Series[str] = IpV4AddressField(nullable=True)
    org: Series[str] = pa.Field(str_length={"min_value": 1}, nullable=True)
    dst_country: Series[str] = Iso3166CountryField(nullable=True)
    dst_lat: Series[float] = LatitudeField(nullable=True)
    dst_lon: Series[float] = LongitudeField(nullable=True)


_MergeInIndexersInfoInputDataFrame = DataFrame[_MergeInIndexersInfoInputSchema]
_MergeInIndexersInfoMixinDataFrame = DataFrame[_MergeInIndexersInfoMixinSchema]


@overload
def merge_in_indexers_info(
    combined_queries: _MergeInIndexersInfoInputDataFrame,
    indexers: IndexersDataFrame,
) -> _MergeInIndexersInfoMixinDataFrame:
    pass  # Type hint for the function signature


@overload
def merge_in_indexers_info(
    combined_queries: DataFrame,
    indexers: IndexersDataFrame,
) -> DataFrame:
    pass  # Type hint for the function signature


def merge_in_indexers_info(
    combined_queries,
    indexers,
):
    """
    Merge in the indexers information into the combined query data.

    This function performs a left merge operation, combining data from the combined query results
    with the unique URL and indexer information. The merge is based on the 'indexer', 'day_partition',
    and 'url' columns.

    :param combined_queries: DataFrame containing the combined query results.
    :param indexers: DataFrame containing unique URLs and indexers information.

    :returns: A new DataFrame resulting from the left merge of the input DataFrames.
    """
    # Rename destination (indexer) location columns
    right_df = indexers.rename(
        columns={
            "country": "dst_country",
            "latitude": "dst_lat",
            "longitude": "dst_lon",
        },
    )

    # Merge the combined query data with the indexers information
    dataframe = pd.merge(
        left=combined_queries,
        right=right_df,
        on=["indexer", "url"],
        how="left",
    )

    return dataframe


class _MergeInQueryGeolocationInputSchema(pa.DataFrameModel):
    """
    Schema for the merge_in_query_geolocation_info function input DataFrame.
    """

    query_id: Series[str] = QueryIdField()


class _MergeInQueryGeolocationMixinSchema(pa.DataFrameModel):
    """
    Schema for the combined query data with additional columns for geolocation information.

    This is an intermediate datatype result of merge_in_query_geolocation_info function.
    """

    IATA_code: Series[str] = IataCodeField()
    src_country: Series[str] = Iso3166CountryField(nullable=True)
    src_lat: Series[float] = LatitudeField(nullable=True)
    src_lon: Series[float] = LongitudeField(nullable=True)


_MergeInQueryGeolocationInputDataFrame = DataFrame[_MergeInQueryGeolocationInputSchema]
_MergeInQueryGeolocationMixinDataFrame = DataFrame[_MergeInQueryGeolocationMixinSchema]


@overload
def merge_in_query_geolocation_info(
    combined_queries: _MergeInQueryGeolocationInputDataFrame,
    iata_info: Optional[IataInfoDataFrame] = None,
) -> _MergeInQueryGeolocationMixinDataFrame:
    pass  # Type hint for the function signature


@overload
def merge_in_query_geolocation_info(
    combined_queries: DataFrame,
    iata_info: Optional[IataInfoDataFrame] = None,
) -> DataFrame:
    pass  # Type hint for the function signature


def merge_in_query_geolocation_info(
    combined_queries,
    iata_info=None,
):
    """
    Merge in query (IATA code-based) geolocation information into combined query data.

    This function extracts the IATA code from the 'query_id' column of the combined query
    DataFrame and merges in the corresponding geolocation information from the IATA DataFrame.
    The merge is performed on the 'IATA_code' column.

    :param combined_queries: DataFrame containing combined queries data.
    :param iata_info: DataFrame containing IATA code information (optional).

    :returns: A new DataFrame resulting from the right merge of the input DataFrames.
    """
    # Extract the IATA code from the 'query_id' column of a DataFrame
    combined_queries["IATA_code"] = combined_queries["query_id"].str[-3:]

    # Merge in the IATA information
    if iata_info is None:
        iata_info = get_iata_geolocation_info()

    # Rename source (IATA) location columns
    right_df = iata_info.rename(
        columns={
            "country": "src_country",
            "latitude": "src_lat",
            "longitude": "src_lon",
        },
    )

    # Merge the combined query data with the IATA information
    dataframe = pd.merge(
        left=combined_queries,
        right=right_df,
        on="IATA_code",
        how="left",
    )

    return dataframe


def haversine_vectorized(
    lon1: pd.Series,
    lat1: pd.Series,
    lon2: pd.Series,
    lat2: pd.Series,
) -> Series[float]:
    """
    Calculate the spherical distances between two sets of coordinates using the Haversine formula.

    This function computes distances between multiple pairs of points on Earth's surface,
    treating the Earth as a sphere. It uses a vectorized implementation for efficiency.

    :param lon1: Longitudes of the first set of points
    :param lat1: Latitudes of the first set of points
    :param lon2: Longitudes of the second set of points
    :param lat2: Latitudes of the second set of points

    :returns: An array of distances in miles between each pair of points
    """
    lon1, lat1, lon2, lat2 = np.radians([lon1, lat1, lon2, lat2])
    dlon = lon2 - lon1
    dlat = lat2 - lat1
    a = np.sin(dlat / 2) ** 2 + np.cos(lat1) * np.cos(lat2) * np.sin(dlon / 2) ** 2
    c = 2 * np.arcsin(np.sqrt(a))
    r = 3956  # Radius of earth in miles

    return c * r


class _CalculateDistancesInputSchema(pa.DataFrameModel):
    """
    Schema for the calculate_distances function input DataFrame.
    """

    src_lat: Series[float] = LatitudeField(nullable=True)
    src_lon: Series[float] = LongitudeField(nullable=True)
    dst_lat: Series[float] = LatitudeField(nullable=True)
    dst_lon: Series[float] = LongitudeField(nullable=True)


class _CalculateDistancesMixinSchema(pa.DataFrameModel):
    """
    Schema for the combined query data with an additional column for calculated distances.
    """

    src_lat: Series[float] = LatitudeField(nullable=True)
    src_lon: Series[float] = LongitudeField(nullable=True)
    dst_lat: Series[float] = LatitudeField(nullable=True)
    dst_lon: Series[float] = LongitudeField(nullable=True)
    distance_miles: Series[float] = pa.Field(ge=0, nullable=True)


_CalculateDistancesInputDataFrame = DataFrame[_CalculateDistancesInputSchema]
_CalculateDistancesMixinDataFrame = DataFrame[_CalculateDistancesMixinSchema]


@overload
def calculate_distances(
    data: _CalculateDistancesInputDataFrame,
) -> _CalculateDistancesMixinDataFrame:
    pass  # Type hint for the function signature


@overload
def calculate_distances(
    data: DataFrame,
) -> DataFrame:
    pass  # Type hint for the function signature


def calculate_distances(data):
    """
    Calculate the spherical distances between origin and destination coordinates.

    This function applies the Haversine formula to compute the distance between each pair
    of origin and destination coordinates in the input DataFrame and rounds the distances
    to the nearest multiple of 250 miles to simplify distance measurements.

    :param data: Input DataFrame containing columns:
        - 'src_lon': Longitude of the origin location
        - 'src_lat': Latitude of the origin location
        - 'dst_lon': Longitude of the destination location
        - 'dst_lat': Latitude of the destination location

    :returns: The input DataFrame minus the original coordinate columns, but with an additional 'distance_miles' column
              containing the calculated distances in miles.
    """
    data["distance_miles"] = haversine_vectorized(
        data["src_lon"],
        data["src_lat"],
        data["dst_lon"],
        data["dst_lat"],
    )

    # Round the distance to the nearest multiple of 250 miles to simplify distance measurements
    data["distance_miles"] = data["distance_miles"].apply(
        lambda val: round(val / 250.0) * 250.0 if pd.notna(val) else val
    )

    return data


class _FilterSuccessfulQueriesInputSchema(pa.DataFrameModel):
    """
    Schema for the filter_successful_queries function input DataFrame.
    """

    status: Series[str] = pa.Field(str_length={"min_value": 1}, nullable=True)


class _FilterSuccessfulQueriesMixinSchema(pa.DataFrameModel):
    """
    Schema for the filtered DataFrame containing only successful queries.
    """

    status: Series[str] = pa.Field(isin=[REQUEST_STATUS_OK])


_FilterSuccessfulQueriesInputDataFrame = DataFrame[_FilterSuccessfulQueriesInputSchema]
_FilterSuccessfulQueriesMixinDataFrame = DataFrame[_FilterSuccessfulQueriesMixinSchema]


@overload
def filter_successful_queries(
    data: _FilterSuccessfulQueriesInputDataFrame,
) -> _FilterSuccessfulQueriesMixinDataFrame:
    pass  # Type hint for the function signature


@overload
def filter_successful_queries(
    data: DataFrame,
) -> DataFrame:
    pass  # Type hint for the function signature


def filter_successful_queries(data):
    """
    Filter the DataFrame to include only rows where the status is '200 OK'.

    This function creates a new DataFrame containing only the rows from the input
    DataFrame where the 'status' column has the value '200 OK'.

    :param data: Input DataFrame containing a 'status' column.
    :returns: A new DataFrame with only the rows where status is '200 OK'.
    """
    dataframe = data[data["status"] == REQUEST_STATUS_OK].copy()

    return dataframe


def iterative_filter(
    df: pd.DataFrame,
    min_deployment_indexers: int,
    min_deployments_per_indexer: int,
    min_queries_per_indexer: int,
    min_queries_per_deployment: int,
) -> pd.DataFrame:
    """
    Iteratively filter a DataFrame based on multiple criteria related to deployments, indexers, and queries.

    This function applies a series of filters to the input DataFrame, removing rows that don't meet
    the specified criteria. It continues to apply these filters iteratively until no further changes occur.

    Note:
    - The filtering process is iterative and continues until the DataFrame size stabilizes.
    - If the filtering results in an empty DataFrame, an empty DataFrame is returned.

    :param df: Input DataFrame containing columns: 'deployment_hash', 'indexer', 'query_id'.
    :param min_deployment_indexers: Minimum number of indexers required for each deployment.
    :param min_deployments_per_indexer: Minimum number of deployments required for each indexer.
    :param min_queries_per_indexer: Minimum number of queries required for each indexer.
    :param min_queries_per_deployment: Minimum number of queries required for each deployment.
    :returns: Filtered DataFrame meeting all specified criteria.
    """
    while True:
        initial_len = len(df)

        # Ensure deployments have at least `min_deployment_indexers` indexers
        indexer_per_deployment = df.groupby("deployment_hash")["indexer"].nunique()
        df = df[
            df["deployment_hash"].map(indexer_per_deployment) >= min_deployment_indexers
        ]

        # Ensure indexers serve at least `min_deployments_per_indexer` deployments
        deployment_per_indexer = df.groupby("indexer")["deployment_hash"].nunique()
        df = df[
            df["indexer"].map(deployment_per_indexer) >= min_deployments_per_indexer
        ]

        # Ensure indexers serve at least `min_queries_per_indexer` unique queries
        queries_per_indexer = df.groupby("indexer")["query_id"].nunique()
        df = df[df["indexer"].map(queries_per_indexer) >= min_queries_per_indexer]

        # Ensure deployments have at least `min_queries_per_deployment` queries
        query_counts_per_deployment = df.groupby("deployment_hash").size()
        df = df[
            df["deployment_hash"].map(query_counts_per_deployment)  # type: ignore
            >= min_queries_per_deployment
        ]

        # Check if no change in DataFrame size, else run the loop again
        if len(df) == initial_len:
            break

    return pd.DataFrame(df)


def strategic_sample(
    df: pd.DataFrame, target_rows_per_subgraph: int
) -> Tuple[pd.DataFrame, int]:
    """
    Sample query_id's in a way that creates balanced representation across indexers on each subgraph.
    The function adds a new column ('sampled_query_id') with some values set to None.

    Note:
    - The function does not reduce the size of the input DataFrame. It only marks sampled rows.
    - The actual number of sampled rows in the whole DataFrame will be greater than target_rows_per_subgraph,
      the number of sampled rows should approximate to 'target_rows' (not passed here as a parameter but defined
      inside the iisa.py) as sampling rows is done separately for each (deployment_hash, indexer) combination.
    - Each deployment_hash is sampled for (target_rows_per_subgraph / number_of_indexers) rows.
    - The function aims for balance: it tries to sample an equal number of rows uniformly for each
      indexer within a deployment_hash, subject to the calculated or provided cap for each deployment_hash.

    :param df: The DataFrame to sample.
    :param target_rows_per_subgraph: The number of rows (queries) to target for each deployment_hash.

    :returns: A tuple containing two elements:
        - The input DataFrame with an additional 'sampled_query_id' column.
          This column contains the sampled query IDs where applicable, and None for non-sampled rows.
        - The square root of the number of sampled query_ids, intended to inform the number of buckets for
          subsequent hashing operations.
    """
    if df.empty:
        df["sampled_query_id"] = pd.Series(dtype="float64")
        return df, 0

    # Calculate number of unique indexers per subgraph.
    # Then calculate how many queries to sample for each indexer, subgraph combination.
    # In the lambda function, x represents the number of unique indexers for a particular deployment_hash.
    indexers_per_subgraph = df.groupby("deployment_hash")["indexer"].nunique()
    cap_per_indexer = indexers_per_subgraph.map(
        lambda x: target_rows_per_subgraph // x if x else 0
    ).to_dict()

    # Create a DataFrame that contains the info above
    query_counts = (
        df.groupby(["deployment_hash", "indexer"])["query_id"]
        .agg(lambda x: list(x.unique()))
        .reset_index(name="unique_query_ids")
    )
    query_counts["cap"] = query_counts["deployment_hash"].map(cap_per_indexer)

    # Then sample the query_id's associated with each indexer, subgraph combination
    def sample_queries(query_ids, cap):
        query_ids = (
            list(np.concatenate(query_ids))
            if isinstance(query_ids[0], list)
            else query_ids
        )
        return np.random.choice(query_ids, size=min(len(query_ids), cap), replace=False)

    # Apply sampling function
    query_counts["sampled_query_id_list"] = query_counts.apply(
        lambda x: sample_queries(x["unique_query_ids"], x["cap"]), axis=1
    )

    # Filter the df with the sampled id's
    # x represents each individual query ID from the df["query_id"] Series
    sampled_ids = set(np.concatenate(query_counts["sampled_query_id_list"].values))
    df["sampled_query_id"] = df["query_id"].apply(
        lambda x: x if x in sampled_ids else None
    )

    # Take the square root of the number of sampled id's to inform the number of buckets to hash mod the query into.
    integer_root = int(np.sqrt(len(sampled_ids)))

    return df, integer_root


def hash_sampled_queries(df: pd.DataFrame, integer_root: int) -> pd.DataFrame:
    """
    Hash the sampled query IDs to create a new column with hashed values.

    This function takes a DataFrame with a 'sampled_query_id' column and creates a new column
    'sampled_query_id_hashed_mod_integer_root' containing the hash of each sampled query ID
    modulo the provided integer root.

    :param df: Input DataFrame containing a 'sampled_query_id' column.
    :param integer_root: The modulus to apply to the hash values.
    :returns: A copy of the input DataFrame with an additional column
              'sampled_query_id_hashed_mod_integer_root' containing the hashed values.
    """
    # Create a copy of the input DataFrame
    result_df = df.copy()

    result_df.loc[
        result_df["sampled_query_id"].notna(),
        "sampled_query_id_hashed_mod_integer_root",
    ] = result_df["sampled_query_id"].apply(lambda x: hash(x) % integer_root)

    return result_df


def perform_latency_linear_regression(
    df: pd.DataFrame, predictor: list[str], categorical: list[str], numeric: list[str]
) -> Tuple[pd.DataFrame, pd.DataFrame]:
    """
    Perform latency linear regression analysis on the given data.

    This function orchestrates the entire latency linear regression process, including data preprocessing,
    model fitting, prediction, and result analysis. It also calculates robust normalized coefficients
    for indexer rankings.

    :param df: The data to perform latency linear regression on.
    :param predictor: List of column names to be used as the dependent variable(s).
    :param categorical: List of column names containing categorical features.
    :param numeric: List of column names containing numeric features.

    :returns: A tuple containing three elements:
        - The original DataFrame with additional columns from latency linear regression results.
        - DataFrame containing indexer rankings - based on robust normalized coefficients.
        - DataFrame containing results [Variable, Latency Coefficient, Standard Error, p-value]
          from the latency linear regression.
    """
    # Preprocess the data    # Preprocess the data
    x, y, preprocessor = _latency_linear_regression_preprocess_data(
        df, predictor, categorical, numeric
    )

    # Perform latency linear regression
    pipeline, y_pred = _latency_linear_regression_create_pipeline(x, y, preprocessor)

    # Analyze the results
    latency_linear_regression_results_df = _latency_linear_regression_analyze_results(
        pipeline, x, y, y_pred
    )

    # Calculate robust normalized coefficients
    # TODO: Use latency_linear_regression_indexer_rankings for pre-filtering which indexers can be assigned
    #       indexing agreements, so in this case we would consider dropping indexers who's ['Robust Normalized
    #       Latency Coefficient + Error Confidence Interval'] is greater than a threshold. Indicating their
    #       performance would not be sufficiently high to justify allocating them an indexing agreement even
    #       via round robin approach.
    latency_linear_regression_indexer_rankings = (
        _latency_linear_regression_calculate_robust_normalized_coefficients(
            latency_linear_regression_results_df
        )
    )

    return (
        latency_linear_regression_indexer_rankings,
        latency_linear_regression_results_df,
    )


def _latency_linear_regression_preprocess_data(
    df: pd.DataFrame, predictor: list[str], categorical: list[str], numeric: list[str]
) -> Tuple[pd.DataFrame, pd.DataFrame, ColumnTransformer]:
    """
    Preprocess data for latency linear regression by encoding categorical variables and scaling numeric variables.

    This function prepares the input data for latency linear regression by separating features and target variables,
    and applying appropriate preprocessing techniques to categorical and numeric features.

    :param df: The input DataFrame containing all variables.
    :param predictor: List of column names to be used as the dependent variable(s).
    :param categorical: List of column names containing categorical features.
    :param numeric: List of column names containing numeric features.

    :returns: A tuple containing three elements:
        - Preprocessed feature DataFrame (X).
        - Target variable DataFrame (y).
        - The preprocessor object used for transforming the data.
    """
    model_columns = categorical + numeric

    # Define features (X) and target (y)
    x = df[model_columns]
    y = df[predictor]

    # Use a Column transformer to apply OneHotEncoder to categorical data and StandardScaler to numeric data.
    preprocessor = ColumnTransformer(
        transformers=[
            (
                "one_hot",
                OneHotEncoder(handle_unknown="ignore", drop="first"),
                categorical,
            ),
            ("scaler", StandardScaler(), numeric),
        ],
        remainder="passthrough",
    )

    return x, y, preprocessor


def _latency_linear_regression_create_pipeline(
    x: pd.DataFrame, y: pd.DataFrame, preprocessor: ColumnTransformer
) -> Tuple[Pipeline, np.ndarray]:
    """
    Perform latency linear regression using preprocessed data.

    This function creates a regression pipeline that includes the preprocessor and a linear regression model,
    fits the pipeline to the data, and generates predictions.

    :param x: The feature DataFrame.
    :param y: The target variable DataFrame.
    :param preprocessor: The preprocessor object for transforming the features.

    :returns: A tuple containing two elements:
        - The fitted regression pipeline.
        - The predicted values (y_pred)."""
    # Create regression pipeline
    pipeline = Pipeline(
        [("preprocessor", preprocessor), ("regressor", LinearRegression())],
        memory=None,
    )

    # Fit pipeline & Use pipeline to predict Y
    pipeline.fit(x, y)
    y_pred = pipeline.predict(x)

    return pipeline, y_pred


def _latency_linear_regression_analyze_results(
    pipeline: Pipeline, x: pd.DataFrame, y: pd.DataFrame, y_pred: np.ndarray
) -> pd.DataFrame:
    """
    Analyze the results of the latency linear regression.

    This function computes various statistical measures to evaluate the performance of the regression model,
    including coefficients, standard errors, and p-values for each feature.

    :param pipeline: The fitted regression pipeline.
    :param x: The feature DataFrame.
    :param y: The actual target variable DataFrame.
    :param y_pred: The predicted values from the model.

    :returns: A DataFrame containing the following columns for each feature:
        - 'Variable': Name of the feature
        - 'Latency Coefficient': Estimated coefficient
        - 'Standard Error': Standard error of the coefficient
        - 'p-value': p-value for the coefficient
    """
    # Calculate the mean_squared_error
    mse = mean_squared_error(y, y_pred)

    # Extract feature names and coefficients from the regression pipeline
    feature_names = pipeline.named_steps["preprocessor"].get_feature_names_out()
    coefficients = pipeline.named_steps["regressor"].coef_

    # Ensure coefficients are a flat array
    if coefficients.ndim > 1:
        coefficients = coefficients.flatten()

    # Calculate standard error of each coefficient
    x_transformed = pipeline.named_steps["preprocessor"].transform(x)
    xtx_inv = pinv(
        np.dot(x_transformed.T, x_transformed) + np.eye(x_transformed.shape[1]) * 1.0
    )
    var_covar_matrix = mse * xtx_inv
    std_errors = np.sqrt(np.diag(var_covar_matrix))

    # Calculate significance of latency linear regression coefficients
    deg_freedom = len(y) - len(coefficients)
    t_scores = coefficients / std_errors
    p_values = [2 * (1 - t.cdf(abs(t_score), deg_freedom)) for t_score in t_scores]

    # Create results_df (latency_linear_regression_results_df)
    latency_linear_regression_results_df = pd.DataFrame(
        {
            "Variable": feature_names,
            "Latency Coefficient": coefficients,
            "Standard Error": std_errors,
            "p-value": p_values,
        }
    )

    return latency_linear_regression_results_df


def _latency_linear_regression_calculate_robust_normalized_coefficients(
    results_df: pd.DataFrame,
) -> pd.DataFrame:
    """
    Calculate robust normalized coefficients for indexer rankings based on latency linear regression results.

    This function processes the latency linear regression results to create a ranking of indexers based on
    their coefficients, adjusting for statistical uncertainty and normalizing the results.

    :param results_df: DataFrame containing latency linear regression results, including coefficients
                       and standard errors for each variable.
    :returns: A DataFrame with columns:
        - 'indexer': Identifier for each indexer
        - 'Latency Coefficient': Original latency linear regression coefficient
        - 'Standard Error': Standard error of the coefficient
        - 'p-value': p-value of the coefficient
        - 'Latency Coefficient + Error Confidence Interval': Latency Coefficient adjusted by adding x times the standard error
        - 'Robust Normalized Latency Coefficient + Error Confidence Interval': Normalized version of the adjusted coefficient
    """
    # Extract indexer coefficients
    indexer_rankings = results_df[
        (results_df["Variable"].str.startswith("one_hot__indexer_"))
        & (~results_df["Variable"].str.startswith("one_hot__indexer_network_"))
    ].sort_values(by="Latency Coefficient")

    # Reset the index and remove the old index column for a clean, sequential index
    indexer_rankings.reset_index(inplace=True)
    indexer_rankings.drop(columns=["index"], inplace=True)

    # Drop one_hot__indexer_ from coefficient names
    indexer_rankings["Variable"] = indexer_rankings["Variable"].str.replace(
        "one_hot__indexer_", ""
    )

    # Rename columns appropriately
    indexer_rankings.rename(columns={"Variable": "indexer"}, inplace=True)

    # Drop nan's
    indexer_rankings.dropna(
        subset=["Latency Coefficient", "Standard Error", "p-value"], inplace=True
    )

    # Calculate the latency coefficient + add a suitable error band on top.
    indexer_rankings["Latency Coefficient + Error Confidence Interval"] = (
        indexer_rankings["Latency Coefficient"]
        + LATENCY_COEFFICIENT_STANDARD_ERROR_MULTIPLIER
        * indexer_rankings["Standard Error"]
    )

    # Calculate the median and IQR
    median_val = indexer_rankings[
        "Latency Coefficient + Error Confidence Interval"
    ].median()
    q1 = indexer_rankings["Latency Coefficient + Error Confidence Interval"].quantile(
        0.25
    )
    q3 = indexer_rankings["Latency Coefficient + Error Confidence Interval"].quantile(
        0.75
    )
    iqr_val = q3 - q1

    # Normalize the 'Latency Coefficient + Error Confidence Interval' using median and IQR
    indexer_rankings[
        "Robust Normalized Latency Coefficient + Error Confidence Interval"
    ] = (
        indexer_rankings["Latency Coefficient + Error Confidence Interval"] - median_val
    ) / iqr_val

    return indexer_rankings


def calculate_indexer_success_rate(df: pd.DataFrame) -> pd.DataFrame:
    """
    Calculate the success rate for each indexer based on query status.

    This function computes the proportion of successful queries (status '200 OK' or 'Unavailable(MissingBlock)')
    for each indexer in the dataset.

    :param df: Input DataFrame containing 'indexer' and 'status' columns.
    :returns: A DataFrame with columns:
        - 'indexer': Unique identifier for each indexer
        - 'average_status': The proportion of successful queries for each indexer (range 0 to 1)
    """
    df_filtered = df[["indexer", "status"]].copy()
    df_filtered["status_numeric"] = df_filtered["status"].apply(
        lambda x: 1
        if x in [REQUEST_STATUS_OK, REQUEST_STATUS_UNAVAILABLE_MISSING_BLOCK]
        else 0
    )
    indexer_success_rate = (
        df_filtered.groupby("indexer")
        .agg(average_status=("status_numeric", "mean"))
        .reset_index()
    )

    # Sorting by indexer name as a tie-breaker when success rates are equal.
    return indexer_success_rate.sort_values(
        by=["average_status", "indexer"], ascending=[True, True]
    )


def calculate_indexer_uptime(
    df: pd.DataFrame, threshold_seconds: Optional[int] = 120
) -> pd.DataFrame:
    """
    Calculate the indexer uptime based on query response statuses and timestamps.

    This function computes two types of uptime metrics for each indexer:
    1. Full uptime: Considers the entire time range between queries.
    2. Restricted uptime: Limits the considered time between queries to a 'threshold' e.g. 120 seconds.

    The uptime calculation process involves:
    1. Determining the midpoint between consecutive timestamps for each indexer.
    2. Considering an indexer as 'up' if the status is '200 OK' or 'Unavailable(MissingBlock)'.
    3. Calculating the duration between midpoints infront and after a specific query response when the indexer is 'up'.
    4. Summing these durations to get the total uptime (seconds) for each indexer.
    5. Comparing the uptime to the total observed time to calculate the percentage uptime.

    The restricted uptime calculation differs in the following ways:
    - Both the restricted uptime and the total observed time are capped at the threshold for each interval.
    - This results in a separate, tailored calculation where both the numerator (restricted uptime)
      and denominator (observed time) are adjusted based on the threshold.
    - The restricted uptime percentage may differ significantly from the full uptime
      percentage, especially when there are large gaps between queries.

    :param df: Input DataFrame containing 'indexer', 'timestamp', and 'status' columns.
    :param threshold_seconds: Maximum time gap to consider for restricted uptime calculation. Defaults to 120 seconds.
    :returns: A DataFrame with columns:
        - 'indexer': Unique identifier for each indexer
        - 'observed_duration_restricted': Total observed time within the threshold
        - 'uptime_duration_restricted': Calculated uptime within the threshold
        - '% up_x': Percentage uptime based on restricted calculation
        - 'observed_duration_full': Total observed time without restrictions
        - 'uptime_duration_full': Calculated uptime without restrictions
        - '% up_y': Percentage uptime based on full calculation
    """
    df_copy = df.copy()
    df_copy["timestamp"] = pd.to_datetime(df_copy["timestamp"])
    df_copy.sort_values(by=["indexer", "timestamp"], inplace=True)

    # Calculate next and previous timestamps for each query
    df_copy["next_timestamp"] = df_copy.groupby("indexer")["timestamp"].shift(-1)
    df_copy["previous_timestamp"] = df_copy.groupby("indexer")["timestamp"].shift(1)

    # Calculate the seconds to the next/previous timestamps.
    df_copy["gap_to_next_query"] = (
        df_copy["next_timestamp"] - df_copy["timestamp"]
    ).dt.total_seconds()
    df_copy["gap_to_previous_query"] = (
        df_copy["timestamp"] - df_copy["previous_timestamp"]
    ).dt.total_seconds()

    # Set next_midpoint as the current timestamp plus half the gap to the next query
    # If a query represents the final query in the data for the indexer then next_midpoint is just equal to timestamp
    df_copy["next_midpoint"] = df_copy["timestamp"] + pd.to_timedelta(
        df_copy["gap_to_next_query"] / 2, unit="s"
    )
    df_copy["next_midpoint"] = df_copy["next_midpoint"].fillna(df_copy["timestamp"])

    # Set previous_midpoint as the current timestamp minus half the gap to the prior query
    # If a query represents the first query in the data for the indexer then previous_midpoint is just equal to timestamp
    df_copy["previous_midpoint"] = df_copy["timestamp"] - pd.to_timedelta(
        df_copy["gap_to_previous_query"] / 2, unit="s"
    )
    df_copy["previous_midpoint"] = df_copy["previous_midpoint"].fillna(
        df_copy["timestamp"]
    )

    # Use query response status to inform weather an indexer is online/offline.
    df_copy["is_up"] = (df_copy["status"] == "200 OK") | (
        df_copy["status"] == "Unavailable(MissingBlock)"
    )

    # Calculate uptime durations using next/prior midpoints, when the indexer was up
    df_copy["uptime_duration_full"] = (
        (df_copy["next_midpoint"] - df_copy["previous_midpoint"])
        .dt.total_seconds()
        .where(df_copy["is_up"], 0)
    )
    df_copy["uptime_duration_restricted"] = np.minimum(
        (df_copy["next_midpoint"] - df_copy["previous_midpoint"])
        .dt.total_seconds()
        .where(df_copy["is_up"], 0),
        threshold_seconds,  # type: ignore
    )

    # Calculate observed durations using next/prior midpoints
    df_copy["observed_duration_full"] = (
        df_copy["next_midpoint"] - df_copy["previous_midpoint"]
    ).dt.total_seconds()
    df_copy["observed_duration_restricted"] = np.minimum(
        (df_copy["next_midpoint"] - df_copy["previous_midpoint"]).dt.total_seconds(),
        threshold_seconds,  # type: ignore
    )

    # Save each indexer's uptime
    uptime_duration_full = df_copy.groupby("indexer")["uptime_duration_full"].sum()
    uptime_duration_restricted = df_copy.groupby("indexer")[
        "uptime_duration_restricted"
    ].sum()

    # Save each indexers total observed time.
    observed_duration_full = df_copy.groupby("indexer")["observed_duration_full"].sum()
    observed_duration_restricted = df_copy.groupby("indexer")[
        "observed_duration_restricted"
    ].sum()

    # Merge and Calculate "% up" for the "full" version
    merged_uptime_full = pd.merge(
        observed_duration_full, uptime_duration_full, on="indexer", how="left"
    ).reset_index()
    merged_uptime_full["% up"] = round(
        merged_uptime_full["uptime_duration_full"]
        / merged_uptime_full["observed_duration_full"]
        * 100,
        3,
    )
    merged_uptime_full = merged_uptime_full.sort_values(by="% up", ascending=False)

    # Merge and Calculate "% up" for the "restricted" version
    merged_uptime_restricted = pd.merge(
        observed_duration_restricted,
        uptime_duration_restricted,
        on="indexer",
        how="left",
    ).reset_index()
    merged_uptime_restricted["% up"] = round(
        merged_uptime_restricted["uptime_duration_restricted"]
        / merged_uptime_restricted["observed_duration_restricted"]
        * 100,
        3,
    )
    merged_uptime_restricted = merged_uptime_restricted.sort_values(
        by="% up", ascending=False
    )

    # Final merge
    # merged_uptime_both['% up_x'] represents merged_uptime_restricted["% up"]
    # merged_uptime_both['% up_y'] represents merged_uptime_full["% up"]
    merged_uptime_both = pd.merge(
        merged_uptime_restricted, merged_uptime_full, on="indexer", how="left"
    )
    return merged_uptime_both


def calculate_indexer_stake_to_fees(
    stake_query_pandas: StakeToFeesDataFrame,
) -> pd.DataFrame:
    """
    Calculate the stake-to-fees ratio and its deviation from the median for each indexer.

    This function processes the results of the stake-to-fees query, computing the
    inter-quartile range (IQR) normalized deviation of each indexer's stake-to-fees ratio
    from the median.

    :param stake_query_pandas: DataFrame containing 'indexer' and 'stake_to_fees' columns.
    :returns: A DataFrame with columns:
        - 'indexer': Indexer identifier
        - 'stake_to_fees': Original stake-to-fees ratio
        - 'stake_to_fees_iqr_deviation': IQR-normalized deviation from the median ratio"""

    stake_to_fees = stake_query_pandas[["stake_to_fees"]].copy()

    median_stake_to_fees = stake_to_fees["stake_to_fees"].median()
    q1 = stake_to_fees["stake_to_fees"].quantile(0.25)
    q3 = stake_to_fees["stake_to_fees"].quantile(0.75)
    iqr = q3 - q1

    # TODO: Use stake_to_fees_iqr_deviation to pre-filter which indexers can be assigned
    #       indexing agreements, so in this case we would consider dropping indexers who's
    #       stake_to_fees_iqr_deviation is below a certain threshold, indicating the level of
    #       economic security they provide for the amount of fees they are collecting is already
    #       particularly low, so assigning them more indexing agreements might not be in our
    #       best interest.

    stake_to_fees["stake_to_fees_iqr_deviation"] = (
        stake_to_fees["stake_to_fees"] - median_stake_to_fees
    ) / iqr

    # Ensure the index is named 'indexer' before resetting
    stake_to_fees.index.name = "indexer"

    # Reset the index to make 'indexer' a column
    stake_to_fees = stake_to_fees.reset_index()

    return stake_to_fees


def aggregate_indexer_info(df: DataFrame) -> pd.DataFrame:
    """
    Aggregate organizational and location information for each indexer.

    This function groups the input DataFrame by indexer and aggregates the 'org' and 'destination_loc'
    information, selecting the most frequent value for each. It also rounds the location coordinates
    to the nearest 20 degrees for privacy and generalization purposes.

    :param df: Input DataFrame containing 'indexer', 'org', and 'destination_loc' columns.

    :returns: An aggregated DataFrame with columns:
        - 'indexer': Unique identifier for each indexer
        - 'org': Most frequent organization associated with the indexer
        - 'dst_lat': Most frequent latitude of the destination location rounded to the nearest 20 degrees
        - 'dst_lon': Most frequent longitude of the destination location rounded to the nearest 20 degrees
    """
    # Group the DataFrame by 'indexer' and calculate the most frequent 'org' and 'destination_loc'
    # for each indexer. The `.mode()[0]` is used to select the first mode in case of multiple modes.
    agg_df = (
        df.groupby("indexer")
        .agg(
            {
                "org": lambda x: x.mode()[0] if not x.mode().empty else np.nan,
                "dst_lat": lambda x: x.mode()[0] if not x.mode().empty else np.nan,
                "dst_lon": lambda x: x.mode()[0] if not x.mode().empty else np.nan,
            }
        )
        .reset_index()
    )

    ## Round the location coordinates to the nearest 20 degrees
    def round_to_nearest_20(x):
        """Round a coordinate to the nearest multiple of 20. If the input is NaN, it is returned as is"""
        return x if pd.isna(x) else round(x / 20) * 20

    agg_df["dst_lat"] = agg_df["dst_lat"].apply(round_to_nearest_20)
    agg_df["dst_lon"] = agg_df["dst_lon"].apply(round_to_nearest_20)

    return agg_df


def merge_and_prepare_dataframes(
    indexer_uptime: pd.DataFrame,
    indexer_rankings: pd.DataFrame,
    agg_df: pd.DataFrame,
    indexer_success_rate: pd.DataFrame,
    stake_to_fees: pd.DataFrame,
) -> pd.DataFrame:
    """
    Merge multiple DataFrames related to indexer performance and prepare a consolidated DataFrame.

    This function combines information from various sources including uptime, rankings,
    organizational data, success rates, and stake-to-fees ratios. It also adds placeholder
    columns for additional metrics.

    :param indexer_uptime: DataFrame with indexer uptime information.
    :param indexer_rankings: DataFrame with indexer rankings.
    :param agg_df: DataFrame with aggregated indexer organizational information.
    :param indexer_success_rate: DataFrame with indexer success rates.
    :param stake_to_fees: DataFrame with stake to fees ratios.

    :returns: A merged DataFrame containing all relevant indexer information.
    """
    # Merge df's together
    merged = pd.merge(indexer_uptime, indexer_rankings, on="indexer", how="left")

    # Drop unnecessary columns
    columns_to_drop = ["observed_duration_full", "uptime_duration_full", "% up_y"]
    columns_to_drop = [col for col in columns_to_drop if col in merged.columns]
    merged = merged.drop(columns=columns_to_drop)

    # Drop rows with no useful data if the columns exist
    columns_to_check = ["Latency Coefficient", "Standard Error", "p-value"]
    existing_columns = [col for col in columns_to_check if col in merged.columns]
    if existing_columns:
        merged = merged.dropna(subset=existing_columns)

    # Merge df's together
    merged = pd.merge(merged, agg_df, on="indexer", how="left")

    # Merge df's together
    merged = pd.merge(merged, indexer_success_rate, on="indexer", how="left")

    # Merge df's together
    merged = pd.merge(merged, stake_to_fees, on="indexer", how="left")

    # Add new columns
    merged["existing_dips_agreements"] = 0
    merged["avg_sync_duration"] = np.nan
    merged["indexing_agreement_acceptance_latency"] = np.nan

    return merged
