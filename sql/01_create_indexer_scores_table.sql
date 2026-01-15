-- Create the pre-computed indexer scores table
-- This table stores daily snapshots of indexer performance metrics
-- IISA fetches the latest snapshot on startup/refresh instead of computing in-container

CREATE TABLE IF NOT EXISTS `graph-mainnet.iisa_data_for_dips.indexer_scores` (
  -- Indexer identification
  indexer STRING NOT NULL,
  url STRING,

  -- Latency metrics (from linear regression)
  -- Lower coefficient = faster response times = better
  lat_lin_reg_coefficient FLOAT64,
  lat_coefficient_std_error FLOAT64,
  lat_coefficient_upper_bound FLOAT64,  -- coefficient + 1.5 * std_error

  -- Normalized latency score (0-1 scale, higher = better)
  -- Uses IQR-based robust normalization
  lat_normalized_score FLOAT64,

  -- Uptime metrics
  uptime_score FLOAT64,  -- Fraction of time available (0-1)
  observed_duration_seconds FLOAT64,
  uptime_duration_seconds FLOAT64,

  -- Success rate
  success_rate FLOAT64,  -- Fraction of 200 OK responses (0-1)

  -- Economic security metrics
  stake_to_fees FLOAT64,
  stake_to_fees_iqr_deviation FLOAT64,

  -- Pre-normalized scores (0-1 scale, higher = better)
  -- These are computed once by the CronJob using IQR-based robust normalization
  -- DataProcessor reads these directly; only existing_dips_agreements is normalized per-request
  norm_uptime_score FLOAT64,
  norm_success_rate FLOAT64,
  norm_stake_to_fees FLOAT64,

  -- Organization/location for decentralization checks
  org STRING,
  dst_lat FLOAT64,
  dst_lon FLOAT64,

  -- Metadata
  computed_at TIMESTAMP NOT NULL,
  query_count INT64,  -- Number of queries used in computation
  num_days INT64      -- Lookback window used
)
PARTITION BY DATE(computed_at)
CLUSTER BY indexer;

-- Index for fast lookups by indexer
-- Note: BigQuery doesn't support traditional indexes, but clustering helps
