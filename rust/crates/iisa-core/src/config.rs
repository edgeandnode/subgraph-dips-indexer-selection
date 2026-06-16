//! Governance-tunable scoring parameters.
//!
//! Every default here is a *starting point for governance*, not a magic number
//! welded into code (which was part of Finding 7's transparency critique). The
//! defaults encode the recommendations from `IISA-AUDIT-FINDINGS.md`.

/// Exponents for the weighted-product score. They should sum to ~1.0; the
/// defaults invert the old capital-heavy split — combined quality (reliability
/// + uptime + latency + freshness) is 0.75, price 0.20, stake-adequacy 0.05
/// (small, because adequacy is a *gate*, not a reward axis).
#[derive(Clone, Debug, PartialEq)]
pub struct Weights {
    pub reliability: f64,
    pub uptime: f64,
    pub latency: f64,
    pub freshness: f64,
    pub price: f64,
    pub stake_adequacy: f64,
}

impl Default for Weights {
    fn default() -> Self {
        Weights {
            reliability: 0.30,
            uptime: 0.20,
            latency: 0.15,
            freshness: 0.10,
            price: 0.20,
            stake_adequacy: 0.05,
        }
    }
}

impl Weights {
    pub fn sum(&self) -> f64 {
        self.reliability
            + self.uptime
            + self.latency
            + self.freshness
            + self.price
            + self.stake_adequacy
    }
}

#[derive(Clone, Debug)]
pub struct ScoringConfig {
    pub weights: Weights,

    // --- Confidence handling (Findings 2 & 5) ---
    /// z-score for the confidence bounds (1.96 ≈ 95%).
    pub z: f64,
    /// Beta(α, β) prior pseudo-counts. With α = β = 1 (Laplace), an indexer with
    /// no history gets a conservative middling estimate — never the 1.0 fill the
    /// Python version applied to zero-fee indexers.
    pub prior_alpha: f64,
    pub prior_beta: f64,

    // --- Uptime (Finding 2: absolute target, not 0.97-of-best) ---
    /// Absolute uptime SLA the lower bound is measured against.
    pub uptime_target: f64,
    /// Seconds per effective uptime "trial" when turning an observed duration
    /// into a confidence sample size.
    pub uptime_bucket_secs: f64,

    // --- Latency ---
    /// Scale (in the regression coefficient's units) at which latency utility
    /// falls to 0.5. Absolute, so one outlier can't compress the field.
    pub latency_scale: f64,

    // --- Freshness (new signal the Python version lacked) ---
    pub freshness_scale_secs: f64,
    /// Hard veto: an indexer further behind the chain head than this is rejected.
    pub max_seconds_behind: f64,

    // --- Price (Finding / Recommendation 3: ceiling-anchored, with a floor) ---
    /// Credible cost-of-service floor, to anchor against races-to-the-bottom.
    pub price_floor: f64,

    // --- Stake adequacy (Finding 1) ---
    /// Collateralisation multiple: stake saturates the signal at `kappa ×
    /// value_at_risk`.
    pub stake_kappa: f64,
    /// Hard floor: below this adequacy the indexer is rejected outright.
    pub min_stake_adequacy: f64,

    // --- Incumbency (Finding 3: hysteresis + dwell, not a +0.50 wall) ---
    /// A challenger must score at least `(1 + delta) ×` the incumbent.
    pub replacement_delta: f64,
    /// …sustained over this many consecutive daily evaluations.
    pub dwell_days: u32,
    /// Score-unit penalty added to the replacement threshold to amortise the
    /// initial-sync cost of churning an indexer.
    pub sync_cost: f64,

    // --- Exploration lane (governance-gated; off by default) ---
    /// Fraction of slots reserved for UCB exploration of unproven indexers.
    pub exploration_fraction: f64,
    /// Trials below which an indexer is "unproven" and eligible for the lane.
    pub proven_trials: f64,

    // --- Decentralisation (Finding 7: ASN/geo diversity, binding) ---
    pub min_distinct_orgs: usize,
    pub min_distinct_regions: usize,
    /// When true, an unsatisfiable diversity requirement is surfaced
    /// (`diversity_satisfied = false`) instead of being silently ignored.
    pub diversity_binding: bool,
    /// Degrees per geo bucket when deriving a region key from lat/lon.
    pub geo_region_degrees: f64,
}

impl Default for ScoringConfig {
    fn default() -> Self {
        ScoringConfig {
            weights: Weights::default(),
            z: 1.96,
            prior_alpha: 1.0,
            prior_beta: 1.0,
            uptime_target: 0.995,
            uptime_bucket_secs: 60.0,
            latency_scale: 500.0,
            freshness_scale_secs: 30.0,
            max_seconds_behind: 300.0,
            price_floor: 0.0,
            stake_kappa: 2.0,
            min_stake_adequacy: 0.25,
            replacement_delta: 0.10,
            dwell_days: 3,
            sync_cost: 0.0,
            exploration_fraction: 0.0,
            proven_trials: 1000.0,
            min_distinct_orgs: 2,
            min_distinct_regions: 2,
            diversity_binding: true,
            geo_region_degrees: 5.0,
        }
    }
}
