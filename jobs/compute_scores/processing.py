"""
Processing logic for computing indexer scores.

This module contains functions extracted and adapted from the IISA codebase
for computing latency regression, uptime, success rate, and stake-to-fees metrics.
"""

import gzip
import json
import logging
import socket
from datetime import date, datetime
from pathlib import Path
from struct import unpack
from typing import Dict, Optional, Tuple
from urllib.parse import urlparse

import airportsdata
import numpy as np
import pandas as pd
import requests
from numpy.linalg import pinv
from scipy.stats import t
from sklearn.compose import ColumnTransformer
from sklearn.linear_model import LinearRegression
from sklearn.metrics import mean_squared_error
from sklearn.pipeline import Pipeline
from sklearn.preprocessing import OneHotEncoder, StandardScaler
from tenacity import (
    retry,
    retry_if_exception_type,
    stop_after_attempt,
    wait_exponential,
)

logger = logging.getLogger(__name__)

# Constants
LATENCY_COEFFICIENT_STANDARD_ERROR_MULTIPLIER = 1.5
REQUEST_STATUS_OK = "200 OK"
REQUEST_STATUS_UNAVAILABLE_MISSING_BLOCK = "Unavailable(MissingBlock)"

# Iterative filter constants
ITERATIVE_FILTER_MIN_DEPLOYMENT_INDEXERS = 2
ITERATIVE_FILTER_MIN_DEPLOYMENTS_PER_INDEXER = 1
ITERATIVE_FILTER_MIN_QUERIES_PER_INDEXER = 250
ITERATIVE_FILTER_MIN_QUERIES_PER_DEPLOYMENT = 250

# Column rename mappings for geolocation data
GEOIP_DST_COLUMN_MAPPING = {
    "country": "dst_country",
    "latitude": "dst_lat",
    "longitude": "dst_lon",
}
GEOIP_SRC_COLUMN_MAPPING = {
    "country": "src_country",
    "latitude": "src_lat",
    "longitude": "src_lon",
}


def compute_all_scores(
    bq_client,
    start_date: date,
    start_ts: str,
    num_days: int,
    target_rows: int,
    ipinfo_auth: str,
) -> pd.DataFrame:
    """
    Compute all indexer scores and return as a DataFrame ready for BigQuery.

    This is the main orchestration function that:
    1. Fetches raw data from BigQuery
    2. Resolves GeoIP for indexers
    3. Runs latency linear regression
    4. Computes uptime, success rate, stake-to-fees
    5. Pre-normalizes static metrics
    6. Returns a DataFrame matching the indexer_scores schema
    """
    # Fetch initial query results to determine sampling
    initial_query_results = bq_client.fetch_initial_query_results(start_date, num_days)
    target_rows_per_subgraph = adjust_rows(initial_query_results, target_rows)

    # Fetch combined query data (~20M rows)
    combined_queries = bq_client.fetch_combined_query_results(
        start_date, num_days, target_rows_per_subgraph
    )

    # Get unique indexers and resolve GeoIP
    indexers_df = resolve_indexer_geoip(combined_queries, ipinfo_auth)

    # Merge indexer info into combined queries
    combined_queries = merge_in_indexers_info(combined_queries, indexers_df)

    # Merge in query geolocation info (IATA codes)
    combined_queries = merge_in_query_geolocation_info(combined_queries)

    # Save data for uptime calculations before filtering
    data_for_uptime = combined_queries[["indexer", "status", "timestamp"]].copy()

    # Calculate distances
    combined_queries = calculate_distances(combined_queries)

    # Filter to successful queries for regression
    combined_queries_filtered = filter_successful_queries(combined_queries)

    # Prepare for regression
    predictor = ["response_time_ms"]
    categorical = ["indexer", "deployment_hash", "indexer_network", "query_id"]
    numeric = ["distance_miles", "fee"]

    filtered_data = combined_queries_filtered[predictor + categorical + numeric]
    filtered_data = filtered_data.dropna(subset=numeric)

    # Apply iterative filtering
    filtered_data = iterative_filter(
        filtered_data,
        ITERATIVE_FILTER_MIN_DEPLOYMENT_INDEXERS,
        ITERATIVE_FILTER_MIN_DEPLOYMENTS_PER_INDEXER,
        ITERATIVE_FILTER_MIN_QUERIES_PER_INDEXER,
        ITERATIVE_FILTER_MIN_QUERIES_PER_DEPLOYMENT,
    )

    # Strategic sampling
    filtered_data, integer_root = strategic_sample(filtered_data, target_rows_per_subgraph)

    # Hash sampled queries
    filtered_data = hash_sampled_queries(filtered_data, integer_root)

    categorical = [
        "indexer",
        "deployment_hash",
        "indexer_network",
        "sampled_query_id_hashed_mod_integer_root",
    ]

    # Perform latency linear regression
    latency_rankings, latency_results = perform_latency_linear_regression(
        filtered_data, predictor, categorical, numeric
    )

    # Calculate other metrics
    indexer_success_rate = calculate_indexer_success_rate(combined_queries)
    indexer_uptime = calculate_indexer_uptime(data_for_uptime)

    # Fetch and calculate stake-to-fees
    stake_to_fees_raw = bq_client.fetch_stake_to_fees(start_ts)
    stake_to_fees = calculate_indexer_stake_to_fees(stake_to_fees_raw)

    # Aggregate indexer info (org, location)
    agg_df = aggregate_indexer_info(combined_queries)

    # Merge all data together
    merged = merge_and_prepare_dataframes(
        indexer_uptime,
        latency_rankings,
        agg_df,
        indexer_success_rate,
        stake_to_fees,
    )

    # Transform to indexer_scores schema with pre-normalized values
    scores_df = transform_to_scores_schema(merged, num_days)

    return scores_df


def transform_to_scores_schema(merged: pd.DataFrame, num_days: int) -> pd.DataFrame:
    """
    Transform the merged DataFrame to match the indexer_scores BigQuery schema.

    Includes pre-normalizing static metrics.
    """
    # Build the output DataFrame
    scores = pd.DataFrame()

    # Indexer identification
    scores["indexer"] = merged["indexer"]
    scores["url"] = merged.get("url", None)

    # Latency metrics
    scores["lat_lin_reg_coefficient"] = merged.get("Latency Coefficient")
    scores["lat_coefficient_std_error"] = merged.get("Standard Error")
    scores["lat_coefficient_upper_bound"] = merged.get(
        "Latency Coefficient + Error Confidence Interval"
    )

    # Compute normalized latency score
    lat_normalized = merged.get("Robust Normalized Latency Coefficient + Error Confidence Interval")
    if lat_normalized is not None:
        # Invert and rescale to 0-1 (lower latency = higher score)
        scores["lat_normalized_score"] = normalize_to_0_1_inverted(lat_normalized)
    else:
        scores["lat_normalized_score"] = None

    # Uptime metrics
    scores["uptime_score"] = merged.get("% up_x", 0) / 100.0  # Convert percentage to 0-1
    scores["observed_duration_seconds"] = merged.get("observed_duration_restricted")
    scores["uptime_duration_seconds"] = merged.get("uptime_duration_restricted")

    # Success rate
    scores["success_rate"] = merged.get("average_status")

    # Economic security metrics
    scores["stake_to_fees"] = merged.get("stake_to_fees")
    scores["stake_to_fees_iqr_deviation"] = merged.get("stake_to_fees_iqr_deviation")

    # Pre-normalized scores
    scores["norm_uptime_score"] = normalize_to_0_1(scores["uptime_score"])
    scores["norm_success_rate"] = normalize_to_0_1(scores["success_rate"])
    scores["norm_stake_to_fees"] = normalize_iqr_to_0_1(
        merged.get("stake_to_fees_iqr_deviation")
    )

    # Organization/location
    scores["org"] = merged.get("org")
    scores["dst_lat"] = merged.get("dst_lat")
    scores["dst_lon"] = merged.get("dst_lon")

    # DIP agreement metrics (placeholders until data sources available - see #15)
    scores["existing_dips_agreements"] = merged.get("existing_dips_agreements", 0)
    scores["avg_sync_duration"] = merged.get("avg_sync_duration")
    scores["indexing_agreement_acceptance_latency"] = merged.get("indexing_agreement_acceptance_latency")

    # Metadata
    scores["computed_at"] = datetime.utcnow()
    scores["query_count"] = len(merged)
    scores["num_days"] = num_days

    return scores


def normalize_to_0_1(series: pd.Series) -> pd.Series:
    """Normalize a series to 0-1 range using min-max scaling."""
    if series is None or series.empty:
        return series
    min_val = series.min()
    max_val = series.max()
    if max_val == min_val:
        return pd.Series([0.5] * len(series))
    return (series - min_val) / (max_val - min_val)


def normalize_to_0_1_inverted(series: pd.Series) -> pd.Series:
    """Normalize and invert (lower = better becomes higher score)."""
    if series is None or series.empty:
        return series
    normalized = normalize_to_0_1(series)
    return 1 - normalized


def normalize_iqr_to_0_1(series: pd.Series) -> pd.Series:
    """
    Normalize IQR deviation to 0-1 range.

    IQR deviations can be negative or positive. We map them so that
    higher stake-to-fees (more economic security) gets a higher score.
    """
    if series is None or series.empty:
        return series
    # Higher deviation = more stake relative to fees = better
    return normalize_to_0_1(series)


def calculate_iqr_deviation(series: pd.Series) -> pd.Series:
    """
    Calculate IQR-based deviation from median.

    Returns (value - median) / IQR for each value in the series.
    Used for robust normalization that's less sensitive to outliers.
    """
    median_val = series.median()
    q1 = series.quantile(0.25)
    q3 = series.quantile(0.75)
    iqr = q3 - q1
    return (series - median_val) / iqr


def resolve_indexer_geoip(combined_queries: pd.DataFrame, ipinfo_auth: str) -> pd.DataFrame:
    """
    Extract unique indexers from query data and resolve their GeoIP information.
    """
    # Get unique indexer/url pairs
    unique_indexers = combined_queries[["indexer", "url"]].drop_duplicates()
    unique_indexers = unique_indexers.dropna(subset=["url"])

    logger.info(f"Resolving GeoIP for {len(unique_indexers)} unique indexers")

    # Create GeoIP resolver
    geoip_cache: Dict[str, dict] = {}

    def resolve_url(url: str) -> dict:
        if url in geoip_cache:
            return geoip_cache[url]

        result = resolve_url_geoip(url, ipinfo_auth)
        geoip_cache[url] = result
        return result

    # Resolve GeoIP for each indexer
    geoip_data = unique_indexers["url"].apply(resolve_url)
    geoip_df = pd.DataFrame(geoip_data.tolist())

    # Combine with indexer info
    indexers_df = pd.concat([unique_indexers.reset_index(drop=True), geoip_df], axis=1)
    indexers_df["indexer_network"] = "arbitrum"

    # Rename columns to match expected schema
    indexers_df = indexers_df.rename(columns=GEOIP_DST_COLUMN_MAPPING)

    return indexers_df


def resolve_url_geoip(url: str, auth: str) -> dict:
    """Resolve GeoIP information for a URL."""
    empty_result = {
        "ip_addr": None,
        "org": None,
        "country": None,
        "latitude": None,
        "longitude": None,
    }

    try:
        parsed = urlparse(url)
        host = parsed.hostname
        if not host:
            return empty_result

        # Resolve hostname to IP
        try:
            _, _, ip_addrs = socket.gethostbyname_ex(host)
            if not ip_addrs:
                return empty_result
            ip_addr = sorted(ip_addrs)[0]
        except socket.gaierror:
            return empty_result

        # Check for private IP
        if is_private_ip(ip_addr):
            return {"ip_addr": ip_addr, **{k: None for k in ["org", "country", "latitude", "longitude"]}}

        # Fetch from ipinfo.io
        if not auth:
            return {"ip_addr": ip_addr, **{k: None for k in ["org", "country", "latitude", "longitude"]}}

        return fetch_ipinfo(ip_addr, auth)

    except Exception as e:
        logger.warning(f"GeoIP resolution failed for {url}: {e}")
        return empty_result


def is_private_ip(ip_addr: str) -> bool:
    """Check if an IP address is private (RFC 1918/3330)."""
    try:
        (ip,) = __import__("struct").unpack(
            "!I", socket.inet_pton(socket.AF_INET, ip_addr)
        )
        private_networks = (
            (0x7F000000, 0xFF000000),  # 127.0.0.0/8
            (0xC0A80000, 0xFFFF0000),  # 192.168.0.0/16
            (0xAC100000, 0xFFF00000),  # 172.16.0.0/12
            (0x0A000000, 0xFF000000),  # 10.0.0.0/8
        )
        return any((ip & mask) == network for network, mask in private_networks)
    except Exception:
        return False


@retry(
    retry=retry_if_exception_type((ConnectionError, requests.exceptions.ConnectionError, socket.timeout)),
    stop=stop_after_attempt(3),
    wait=wait_exponential(multiplier=1, max=30),
)
def fetch_ipinfo(ip_addr: str, auth: str) -> dict:
    """Fetch IP info from ipinfo.io API."""
    try:
        response = requests.get(
            f"https://ipinfo.io/{ip_addr}/json?token={auth}",
            timeout=5,
        )
        response.raise_for_status()

        try:
            data = response.json()
        except requests.exceptions.JSONDecodeError:
            data = json.loads(gzip.decompress(response.content))

        loc = data.get("loc", "")
        lat, lon = None, None
        if loc:
            try:
                lat, lon = map(float, loc.split(","))
            except ValueError:
                pass

        return {
            "ip_addr": data.get("ip"),
            "org": data.get("org"),
            "country": data.get("country"),
            "latitude": lat,
            "longitude": lon,
        }
    except Exception as e:
        logger.warning(f"ipinfo.io request failed: {e}")
        return {
            "ip_addr": ip_addr,
            "org": None,
            "country": None,
            "latitude": None,
            "longitude": None,
        }


# --- Adapted from IISA processing.py ---


def adjust_rows(initial_query_results: pd.DataFrame, target_rows: int) -> int:
    """Adjust rows per group to approximate target total rows."""
    if target_rows < 0:
        raise ValueError("Target rows must be non-negative")

    df = initial_query_results.copy()
    x = 1_000
    df["num_rows_restricted"] = df["num_rows"].clip(upper=x)
    tolerance = target_rows * 0.01
    max_iterations = 1_000

    for _ in range(max_iterations):
        current_sum = df["num_rows_restricted"].sum()
        if target_rows - tolerance <= current_sum <= target_rows + tolerance:
            break
        if current_sum > target_rows:
            x = int(x * 0.99)
        else:
            x = int(x * 1.01)
        df["num_rows_restricted"] = df["num_rows"].clip(upper=x)

    return df["num_rows_restricted"].max()


def merge_in_indexers_info(combined_queries: pd.DataFrame, indexers: pd.DataFrame) -> pd.DataFrame:
    """Merge indexer GeoIP info into combined queries."""
    right_df = indexers.rename(columns=GEOIP_DST_COLUMN_MAPPING)
    return pd.merge(combined_queries, right_df, on=["indexer", "url"], how="left")


def merge_in_query_geolocation_info(combined_queries: pd.DataFrame) -> pd.DataFrame:
    """Merge IATA geolocation info based on query_id suffix."""
    combined_queries["IATA_code"] = combined_queries["query_id"].str[-3:]

    iata_info = load_iata_data()
    right_df = iata_info.rename(columns=GEOIP_SRC_COLUMN_MAPPING)

    return pd.merge(combined_queries, right_df, on="IATA_code", how="left")


def load_iata_data() -> pd.DataFrame:
    """Load IATA airport data from airportsdata package."""
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
    return iata_df


def calculate_distances(data: pd.DataFrame) -> pd.DataFrame:
    """Calculate haversine distances between source and destination."""
    data["distance_miles"] = haversine_vectorized(
        data["src_lon"], data["src_lat"], data["dst_lon"], data["dst_lat"]
    )
    data["distance_miles"] = data["distance_miles"].apply(
        lambda val: round(val / 250.0) * 250.0 if pd.notna(val) else val
    )
    return data


def haversine_vectorized(lon1, lat1, lon2, lat2):
    """Vectorized haversine distance calculation."""
    lon1, lat1, lon2, lat2 = np.radians([lon1, lat1, lon2, lat2])
    dlon = lon2 - lon1
    dlat = lat2 - lat1
    a = np.sin(dlat / 2) ** 2 + np.cos(lat1) * np.cos(lat2) * np.sin(dlon / 2) ** 2
    c = 2 * np.arcsin(np.sqrt(a))
    return c * 3956  # Earth radius in miles


def filter_successful_queries(data: pd.DataFrame) -> pd.DataFrame:
    """Filter to only successful queries (200 OK)."""
    return data[data["status"] == REQUEST_STATUS_OK].copy()


def iterative_filter(
    df: pd.DataFrame,
    min_deployment_indexers: int,
    min_deployments_per_indexer: int,
    min_queries_per_indexer: int,
    min_queries_per_deployment: int,
) -> pd.DataFrame:
    """Iteratively filter data based on minimum thresholds."""
    while True:
        initial_len = len(df)

        # Ensure deployments have minimum indexers
        indexer_per_deployment = df.groupby("deployment_hash")["indexer"].nunique()
        df = df[df["deployment_hash"].map(indexer_per_deployment) >= min_deployment_indexers]

        # Ensure indexers serve minimum deployments
        deployment_per_indexer = df.groupby("indexer")["deployment_hash"].nunique()
        df = df[df["indexer"].map(deployment_per_indexer) >= min_deployments_per_indexer]

        # Ensure indexers serve minimum queries
        queries_per_indexer = df.groupby("indexer")["query_id"].nunique()
        df = df[df["indexer"].map(queries_per_indexer) >= min_queries_per_indexer]

        # Ensure deployments have minimum queries
        query_counts = df.groupby("deployment_hash").size()
        df = df[df["deployment_hash"].map(query_counts) >= min_queries_per_deployment]

        if len(df) == initial_len:
            break

    return pd.DataFrame(df)


def strategic_sample(df: pd.DataFrame, target_rows_per_subgraph: int) -> Tuple[pd.DataFrame, int]:
    """Sample queries to create balanced representation across indexers."""
    if df.empty:
        df["sampled_query_id"] = pd.Series(dtype="float64")
        return df, 0

    indexers_per_subgraph = df.groupby("deployment_hash")["indexer"].nunique()
    cap_per_indexer = indexers_per_subgraph.map(
        lambda x: target_rows_per_subgraph // x if x else 0
    ).to_dict()

    query_counts = (
        df.groupby(["deployment_hash", "indexer"])["query_id"]
        .agg(lambda x: list(x.unique()))
        .reset_index(name="unique_query_ids")
    )
    query_counts["cap"] = query_counts["deployment_hash"].map(cap_per_indexer)

    def sample_queries(query_ids, cap):
        query_ids = (
            list(np.concatenate(query_ids)) if isinstance(query_ids[0], list) else query_ids
        )
        return np.random.choice(query_ids, size=min(len(query_ids), cap), replace=False)

    query_counts["sampled_query_id_list"] = query_counts.apply(
        lambda x: sample_queries(x["unique_query_ids"], x["cap"]), axis=1
    )

    sampled_ids = set(np.concatenate(query_counts["sampled_query_id_list"].values))
    df["sampled_query_id"] = df["query_id"].apply(lambda x: x if x in sampled_ids else None)

    integer_root = int(np.sqrt(len(sampled_ids)))
    return df, integer_root


def hash_sampled_queries(df: pd.DataFrame, integer_root: int) -> pd.DataFrame:
    """Hash sampled query IDs for regression."""
    result_df = df.copy()
    result_df.loc[
        result_df["sampled_query_id"].notna(),
        "sampled_query_id_hashed_mod_integer_root",
    ] = result_df["sampled_query_id"].apply(lambda x: hash(x) % integer_root)
    return result_df


def perform_latency_linear_regression(
    df: pd.DataFrame, predictor: list, categorical: list, numeric: list
) -> Tuple[pd.DataFrame, pd.DataFrame]:
    """Perform latency linear regression analysis."""
    model_columns = categorical + numeric
    x = df[model_columns]
    y = df[predictor]

    preprocessor = ColumnTransformer(
        transformers=[
            ("one_hot", OneHotEncoder(handle_unknown="ignore", drop="first"), categorical),
            ("scaler", StandardScaler(), numeric),
        ],
        remainder="passthrough",
    )

    pipeline = Pipeline([("preprocessor", preprocessor), ("regressor", LinearRegression())])
    pipeline.fit(x, y)
    y_pred = pipeline.predict(x)

    # Analyze results
    mse = mean_squared_error(y, y_pred)
    feature_names = pipeline.named_steps["preprocessor"].get_feature_names_out()
    coefficients = pipeline.named_steps["regressor"].coef_.flatten()

    x_transformed = pipeline.named_steps["preprocessor"].transform(x)
    xtx_inv = pinv(np.dot(x_transformed.T, x_transformed) + np.eye(x_transformed.shape[1]) * 1.0)
    var_covar_matrix = mse * xtx_inv
    std_errors = np.sqrt(np.diag(var_covar_matrix))

    deg_freedom = len(y) - len(coefficients)
    t_scores = coefficients / std_errors
    p_values = [2 * (1 - t.cdf(abs(ts), deg_freedom)) for ts in t_scores]

    results_df = pd.DataFrame({
        "Variable": feature_names,
        "Latency Coefficient": coefficients,
        "Standard Error": std_errors,
        "p-value": p_values,
    })

    # Calculate robust normalized coefficients
    indexer_rankings = results_df[
        (results_df["Variable"].str.startswith("one_hot__indexer_"))
        & (~results_df["Variable"].str.startswith("one_hot__indexer_network_"))
    ].sort_values(by="Latency Coefficient")

    indexer_rankings = indexer_rankings.reset_index(drop=True)
    indexer_rankings["Variable"] = indexer_rankings["Variable"].str.replace("one_hot__indexer_", "")
    indexer_rankings.rename(columns={"Variable": "indexer"}, inplace=True)
    indexer_rankings.dropna(subset=["Latency Coefficient", "Standard Error", "p-value"], inplace=True)

    indexer_rankings["Latency Coefficient + Error Confidence Interval"] = (
        indexer_rankings["Latency Coefficient"]
        + LATENCY_COEFFICIENT_STANDARD_ERROR_MULTIPLIER * indexer_rankings["Standard Error"]
    )

    indexer_rankings["Robust Normalized Latency Coefficient + Error Confidence Interval"] = (
        calculate_iqr_deviation(indexer_rankings["Latency Coefficient + Error Confidence Interval"])
    )

    return indexer_rankings, results_df


def calculate_indexer_success_rate(df: pd.DataFrame) -> pd.DataFrame:
    """Calculate success rate for each indexer."""
    df_filtered = df[["indexer", "status"]].copy()
    df_filtered["status_numeric"] = df_filtered["status"].apply(
        lambda x: 1 if x in [REQUEST_STATUS_OK, REQUEST_STATUS_UNAVAILABLE_MISSING_BLOCK] else 0
    )
    return (
        df_filtered.groupby("indexer")
        .agg(average_status=("status_numeric", "mean"))
        .reset_index()
    )


def calculate_indexer_uptime(df: pd.DataFrame, threshold_seconds: int = 120) -> pd.DataFrame:
    """Calculate indexer uptime based on query timestamps and statuses."""
    df_copy = df.copy()
    df_copy["timestamp"] = pd.to_datetime(df_copy["timestamp"])
    df_copy.sort_values(by=["indexer", "timestamp"], inplace=True)

    df_copy["next_timestamp"] = df_copy.groupby("indexer")["timestamp"].shift(-1)
    df_copy["previous_timestamp"] = df_copy.groupby("indexer")["timestamp"].shift(1)

    df_copy["gap_to_next_query"] = (df_copy["next_timestamp"] - df_copy["timestamp"]).dt.total_seconds()
    df_copy["gap_to_previous_query"] = (df_copy["timestamp"] - df_copy["previous_timestamp"]).dt.total_seconds()

    df_copy["next_midpoint"] = df_copy["timestamp"] + pd.to_timedelta(df_copy["gap_to_next_query"] / 2, unit="s")
    df_copy["next_midpoint"] = df_copy["next_midpoint"].fillna(df_copy["timestamp"])

    df_copy["previous_midpoint"] = df_copy["timestamp"] - pd.to_timedelta(df_copy["gap_to_previous_query"] / 2, unit="s")
    df_copy["previous_midpoint"] = df_copy["previous_midpoint"].fillna(df_copy["timestamp"])

    df_copy["is_up"] = df_copy["status"].isin([REQUEST_STATUS_OK, REQUEST_STATUS_UNAVAILABLE_MISSING_BLOCK])

    df_copy["uptime_duration_full"] = (
        (df_copy["next_midpoint"] - df_copy["previous_midpoint"]).dt.total_seconds().where(df_copy["is_up"], 0)
    )
    df_copy["uptime_duration_restricted"] = np.minimum(
        (df_copy["next_midpoint"] - df_copy["previous_midpoint"]).dt.total_seconds().where(df_copy["is_up"], 0),
        threshold_seconds,
    )

    df_copy["observed_duration_full"] = (df_copy["next_midpoint"] - df_copy["previous_midpoint"]).dt.total_seconds()
    df_copy["observed_duration_restricted"] = np.minimum(
        (df_copy["next_midpoint"] - df_copy["previous_midpoint"]).dt.total_seconds(),
        threshold_seconds,
    )

    # Aggregate by indexer
    uptime_full = df_copy.groupby("indexer")["uptime_duration_full"].sum()
    uptime_restricted = df_copy.groupby("indexer")["uptime_duration_restricted"].sum()
    observed_full = df_copy.groupby("indexer")["observed_duration_full"].sum()
    observed_restricted = df_copy.groupby("indexer")["observed_duration_restricted"].sum()

    merged_restricted = pd.merge(observed_restricted, uptime_restricted, on="indexer", how="left").reset_index()
    merged_restricted["% up"] = round(merged_restricted["uptime_duration_restricted"] / merged_restricted["observed_duration_restricted"] * 100, 3)
    merged_restricted = merged_restricted.sort_values(by="% up", ascending=False)

    merged_full = pd.merge(observed_full, uptime_full, on="indexer", how="left").reset_index()
    merged_full["% up"] = round(merged_full["uptime_duration_full"] / merged_full["observed_duration_full"] * 100, 3)
    merged_full = merged_full.sort_values(by="% up", ascending=False)

    return pd.merge(merged_restricted, merged_full, on="indexer", how="left")


def calculate_indexer_stake_to_fees(stake_query_pandas: pd.DataFrame) -> pd.DataFrame:
    """Calculate stake-to-fees ratio and IQR deviation."""
    stake_to_fees = stake_query_pandas[["stake_to_fees"]].copy()
    stake_to_fees["stake_to_fees_iqr_deviation"] = calculate_iqr_deviation(stake_to_fees["stake_to_fees"])
    stake_to_fees.index.name = "indexer"
    return stake_to_fees.reset_index()


def aggregate_indexer_info(df: pd.DataFrame) -> pd.DataFrame:
    """Aggregate organizational and location info per indexer."""
    def round_to_20(x):
        return x if pd.isna(x) else round(x / 20) * 20

    agg_df = (
        df.groupby("indexer")
        .agg({
            "org": lambda x: x.mode()[0] if not x.mode().empty else np.nan,
            "dst_lat": lambda x: x.mode()[0] if not x.mode().empty else np.nan,
            "dst_lon": lambda x: x.mode()[0] if not x.mode().empty else np.nan,
        })
        .reset_index()
    )

    agg_df["dst_lat"] = agg_df["dst_lat"].apply(round_to_20)
    agg_df["dst_lon"] = agg_df["dst_lon"].apply(round_to_20)

    return agg_df


def merge_and_prepare_dataframes(
    indexer_uptime: pd.DataFrame,
    indexer_rankings: pd.DataFrame,
    agg_df: pd.DataFrame,
    indexer_success_rate: pd.DataFrame,
    stake_to_fees: pd.DataFrame,
) -> pd.DataFrame:
    """Merge all indexer data into a single DataFrame."""
    merged = pd.merge(indexer_uptime, indexer_rankings, on="indexer", how="left")

    columns_to_drop = ["observed_duration_full", "uptime_duration_full", "% up_y"]
    merged = merged.drop(columns=[c for c in columns_to_drop if c in merged.columns])

    columns_to_check = ["Latency Coefficient", "Standard Error", "p-value"]
    existing = [c for c in columns_to_check if c in merged.columns]
    if existing:
        merged = merged.dropna(subset=existing)

    merged = pd.merge(merged, agg_df, on="indexer", how="left")
    merged = pd.merge(merged, indexer_success_rate, on="indexer", how="left")
    merged = pd.merge(merged, stake_to_fees, on="indexer", how="left")

    # Add placeholder columns for metrics not yet populated from data sources
    # These are required by DataProcessor for scoring
    merged["existing_dips_agreements"] = 0
    merged["avg_sync_duration"] = np.nan
    merged["indexing_agreement_acceptance_latency"] = np.nan

    return merged
