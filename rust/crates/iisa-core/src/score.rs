//! Turning one indexer's measurements into a single score via a weighted
//! **product** (Finding 4), behind hard eligibility gates.

use crate::config::{ScoringConfig, Weights};
use crate::domain::{IndexerId, Utility};
use crate::metrics::{
    freshness_utility, latency_utility, price_utility, reliability_ucb, reliability_utility,
    stake_adequacy, uptime_utility,
};

/// The measured inputs for one indexer, for one deployment/chain.
///
/// This is a deliberately richer wire format than the Python push payload: it
/// carries `query_successes`/`query_trials` (not just a ratio) so reliability
/// can be a confidence bound, and `seconds_behind` for the freshness signal.
/// The Python scoring job would need to emit these two extra fields.
#[derive(Clone, Debug)]
pub struct IndexerMetrics {
    pub indexer: IndexerId,

    /// Successful query responses and total query attempts in the window.
    pub query_successes: f64,
    pub query_trials: f64,

    /// Observed uptime ratio in `[0, 1]` and the duration it was observed over.
    pub uptime_ratio: f64,
    pub uptime_observed_secs: f64,

    /// Geo-aware latency regression coefficient CI upper bound (lower better).
    pub latency_ci_upper: f64,

    /// How far behind the chain head the indexer is (lower better).
    pub seconds_behind: f64,

    /// Slashable (self + delegated, minus locked) stake in GRT.
    pub slashable_stake_grt: f64,

    /// Price for the requested chain (GRT per 30 days). `None` means the indexer
    /// cannot serve this chain → ineligible.
    pub price_per_30d_grt: Option<f64>,
    /// Secondary price signal, carried for reporting (not separately weighted).
    pub price_per_billion_entities_grt: Option<f64>,

    /// ASN organisation (from GeoIP) and lat/lon, for diversity scoring.
    pub asn_org: Option<String>,
    pub geo_lat: Option<f64>,
    pub geo_lon: Option<f64>,
}

impl IndexerMetrics {
    /// A neutral starting point; fields are public so callers set what they have.
    pub fn new(indexer: IndexerId) -> Self {
        IndexerMetrics {
            indexer,
            query_successes: 0.0,
            query_trials: 0.0,
            uptime_ratio: 0.0,
            uptime_observed_secs: 0.0,
            latency_ci_upper: 0.0,
            seconds_behind: 0.0,
            slashable_stake_grt: 0.0,
            price_per_30d_grt: None,
            price_per_billion_entities_grt: None,
            asn_org: None,
            geo_lat: None,
            geo_lon: None,
        }
    }
}

/// Per-deployment agreement parameters supplied by the caller (dipper).
#[derive(Clone, Debug)]
pub struct AgreementParams {
    /// Maximum acceptable price (GRT per 30 days) — the price normalisation
    /// anchor, replacing the observed max.
    pub price_ceiling: f64,
    /// Slashable exposure of the agreement, used by stake adequacy.
    pub value_at_risk: f64,
}

/// The utilities that fed the score, kept for transparency/disputability.
#[derive(Clone, Debug)]
pub struct ScoreComponents {
    pub reliability: Utility,
    pub uptime: Utility,
    pub latency: Utility,
    pub freshness: Utility,
    pub price: Utility,
    pub stake_adequacy: Utility,
}

/// The outcome of scoring one indexer.
#[derive(Clone, Debug)]
pub struct Scored {
    pub indexer: IndexerId,
    /// Weighted geometric mean in `[0, 1]`; `0.0` if vetoed or ineligible.
    pub score: f64,
    pub eligible: bool,
    pub reject_reason: Option<String>,
    pub components: ScoreComponents,
    /// Reliability upper-confidence bound for the exploration lane.
    pub reliability_ucb: f64,
    pub org: Option<String>,
    pub geo: Option<(f64, f64)>,
    pub trials: f64,
}

/// Weighted **product** of the component utilities (Finding 4).
///
/// A near-zero utility on any positively-weighted axis drives the whole score
/// to zero — a natural veto that capital cannot buy off, unlike the additive
/// sum the Python version used.
pub fn weighted_product(c: &ScoreComponents, w: &Weights) -> f64 {
    let factors = [
        (c.reliability, w.reliability),
        (c.uptime, w.uptime),
        (c.latency, w.latency),
        (c.freshness, w.freshness),
        (c.price, w.price),
        (c.stake_adequacy, w.stake_adequacy),
    ];
    let mut log_acc = 0.0;
    for (u, weight) in factors {
        if weight <= 0.0 {
            continue;
        }
        let x = u.get();
        if x <= 0.0 {
            return 0.0; // veto
        }
        log_acc += weight * x.ln();
    }
    log_acc.exp().clamp(0.0, 1.0)
}

/// Score one indexer behind the hard eligibility gates.
pub fn score_indexer(m: &IndexerMetrics, cfg: &ScoringConfig, ag: &AgreementParams) -> Scored {
    let reliability = reliability_utility(m.query_successes, m.query_trials, cfg);
    let reliability_ucb = reliability_ucb(m.query_successes, m.query_trials, cfg);
    let uptime = uptime_utility(m.uptime_ratio, m.uptime_observed_secs, cfg);
    let latency = latency_utility(m.latency_ci_upper, cfg);
    let freshness = freshness_utility(m.seconds_behind, cfg);
    let stake = stake_adequacy(m.slashable_stake_grt, ag.value_at_risk, cfg.stake_kappa);

    // --- Hard gates ---
    let mut reject: Option<String> = None;
    let price = match m.price_per_30d_grt {
        None => {
            reject = Some("no price for requested chain".to_string());
            Utility::ZERO
        }
        Some(p) if p > ag.price_ceiling => {
            reject = Some("price above ceiling".to_string());
            Utility::ZERO
        }
        Some(p) => price_utility(p, ag.price_ceiling, cfg.price_floor),
    };
    if reject.is_none() && stake.get() < cfg.min_stake_adequacy {
        reject = Some("insufficient stake adequacy".to_string());
    }
    if reject.is_none() && m.seconds_behind > cfg.max_seconds_behind {
        reject = Some("too far behind chain head".to_string());
    }

    let components = ScoreComponents {
        reliability,
        uptime,
        latency,
        freshness,
        price,
        stake_adequacy: stake,
    };
    let eligible = reject.is_none();
    let score = if eligible {
        weighted_product(&components, &cfg.weights)
    } else {
        0.0
    };

    Scored {
        indexer: m.indexer.clone(),
        score,
        eligible,
        reject_reason: reject,
        components,
        reliability_ucb,
        org: m.asn_org.clone(),
        geo: match (m.geo_lat, m.geo_lon) {
            (Some(a), Some(b)) => Some((a, b)),
            _ => None,
        },
        trials: m.query_trials,
    }
}
