//! Per-metric utility transforms.
//!
//! Every transform here is **absolute**: it maps one indexer's measurement to a
//! `[0, 1]` utility without reference to any other indexer in the field. That
//! single property kills the floating-relative-gate instability of Finding 2
//! and the single-high-bidder compression of the price axis.

use crate::config::ScoringConfig;
use crate::domain::Utility;

/// Wilson score interval bounds for a binomial proportion.
///
/// Returns `(lower, upper)`, both clamped to `[0, 1]`. This is the principled
/// replacement for raw ratios: "100% over 5 trials" yields a far lower bound
/// than "99.9% over 5,000,000", so confidence is rewarded, not gamed.
pub fn wilson_bounds(successes: f64, trials: f64, z: f64) -> (f64, f64) {
    if trials <= 0.0 {
        return (0.0, 1.0);
    }
    let n = trials;
    let phat = (successes / n).clamp(0.0, 1.0);
    let z2 = z * z;
    let denom = 1.0 + z2 / n;
    let centre = phat + z2 / (2.0 * n);
    let margin = z * ((phat * (1.0 - phat) + z2 / (4.0 * n)) / n).sqrt();
    let lower = ((centre - margin) / denom).clamp(0.0, 1.0);
    let upper = ((centre + margin) / denom).clamp(0.0, 1.0);
    (lower, upper)
}

/// Apply the Beta prior pseudo-counts (Finding 5: absence of evidence is
/// handled conservatively, identically across metrics — never filled to 1.0).
fn with_prior(successes: f64, trials: f64, cfg: &ScoringConfig) -> (f64, f64) {
    let s = successes.max(0.0) + cfg.prior_alpha;
    let t = trials.max(0.0) + cfg.prior_alpha + cfg.prior_beta;
    (s, t)
}

/// Reliability (success rate) utility: the *lower* confidence bound. New
/// indexers with few trials get a conservative estimate — not zero, not
/// inflated. Worth the highest weight (Finding 2 fixes the old 0.05).
pub fn reliability_utility(successes: f64, trials: f64, cfg: &ScoringConfig) -> Utility {
    let (s, t) = with_prior(successes, trials, cfg);
    let (lower, _) = wilson_bounds(s, t, cfg.z);
    Utility::clamp(lower)
}

/// Reliability *upper* confidence bound — optimism in the face of uncertainty.
/// Used only by the exploration lane so an unproven indexer can be tried
/// without inflating its exploit score.
pub fn reliability_ucb(successes: f64, trials: f64, cfg: &ScoringConfig) -> f64 {
    let (s, t) = with_prior(successes, trials, cfg);
    let (_, upper) = wilson_bounds(s, t, cfg.z);
    upper
}

/// Uptime utility: confidence-adjusted observed uptime measured against an
/// **absolute** SLA target (Finding 2 — not 0.97 of the best in the field). The
/// observed duration sets the effective sample size, so a long, steady record
/// scores higher than a brief lucky one.
pub fn uptime_utility(ratio: f64, observed_secs: f64, cfg: &ScoringConfig) -> Utility {
    let eff_trials = (observed_secs.max(0.0) / cfg.uptime_bucket_secs).max(0.0);
    let successes = ratio.clamp(0.0, 1.0) * eff_trials;
    let (s, t) = with_prior(successes, eff_trials, cfg);
    let (lower, _) = wilson_bounds(s, t, cfg.z);
    if cfg.uptime_target <= 0.0 {
        return Utility::clamp(lower);
    }
    Utility::clamp(lower / cfg.uptime_target)
}

/// Latency utility from the regression coefficient's CI upper bound (lower is
/// better). Saturating and absolute: 0 → 1.0, `latency_scale` → 0.5.
pub fn latency_utility(ci_upper: f64, cfg: &ScoringConfig) -> Utility {
    let x = ci_upper.max(0.0);
    let scale = if cfg.latency_scale > 0.0 {
        cfg.latency_scale
    } else {
        1.0
    };
    Utility::clamp(1.0 / (1.0 + x / scale))
}

/// Freshness utility from seconds-behind-chain-head (lower is better). The hard
/// veto for badly-stale indexers lives in the score gate; this is the smooth
/// reward within the acceptable band.
pub fn freshness_utility(seconds_behind: f64, cfg: &ScoringConfig) -> Utility {
    let x = seconds_behind.max(0.0);
    let scale = if cfg.freshness_scale_secs > 0.0 {
        cfg.freshness_scale_secs
    } else {
        1.0
    };
    Utility::clamp(1.0 / (1.0 + x / scale))
}

/// Price utility, anchored to the caller's ceiling and a governance floor
/// (Recommendation 3) — NOT the observed max. One high bidder cannot compress
/// everyone else's price differentiation.
pub fn price_utility(price: f64, ceiling: f64, floor: f64) -> Utility {
    if ceiling <= floor {
        // Degenerate band: nothing to differentiate on, stay neutral.
        return Utility::clamp(0.5);
    }
    Utility::clamp((ceiling - price) / (ceiling - floor))
}

/// Stake adequacy (Finding 1): a bounded collateralisation ratio of slashable
/// stake to the *value-at-risk* of the work, saturating at 1.0. Stake is never
/// divided by realised query fees, so earning fees never lowers the score and
/// an idle high-stake address cannot exceed an adequately-staked busy one.
pub fn stake_adequacy(slashable_stake: f64, value_at_risk: f64, kappa: f64) -> Utility {
    if value_at_risk <= 0.0 {
        // Nothing at risk on this agreement → adequacy is trivially met.
        return Utility::ONE;
    }
    let denom = kappa.max(f64::MIN_POSITIVE) * value_at_risk;
    Utility::clamp(slashable_stake / denom)
}
