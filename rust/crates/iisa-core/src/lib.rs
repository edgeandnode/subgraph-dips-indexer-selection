//! Corrected indexer selection algorithm (IISA) for The Graph's DIPs service.
//!
//! This is a Rust rewrite of the Python `src/iisa/indexer_selection.py`, built
//! to fix every finding in `IISA-AUDIT-FINDINGS.md`. The fixes are not comments
//! or conventions — where possible they are enforced by the type system:
//!
//! * **Finding 1 (inverted, global `stake_to_fees`)** → [`metrics::stake_adequacy`]:
//!   a bounded ratio of slashable stake to *value-at-risk*, saturating at 1.0.
//!   Stake is never divided by realised fees, so commercial success is never
//!   penalised and an idle high-stake address cannot max the dominant axis.
//! * **Finding 2 (floating relative quality gate)** → [`metrics::reliability_utility`]
//!   and [`metrics::uptime_utility`]: Wilson lower-confidence bounds against an
//!   *absolute* SLA target, never relative to the best indexer in the field.
//! * **Finding 3 (welded incumbency)** → [`select`]: relative hysteresis plus a
//!   dwell requirement ([`select::ChallengeLedger`]), not a +0.50 absolute wall.
//! * **Finding 4 (additive sum buys off quality)** → [`score::weighted_product`]:
//!   a weighted geometric mean, so a near-zero quality utility vetoes the score.
//! * **Finding 5 (asymmetric NaN fill)** → absence of evidence is treated
//!   conservatively everywhere (Beta prior + LCB); nothing fills *up* to 1.0.
//! * **Finding 6 (determinism)** → [`select::select_indexers`] is a pure function
//!   with stable ordering; identical inputs always yield identical output.
//! * **Finding 7 (org-counting decentralisation, silently bypassed)** → diversity
//!   is ASN/geo-derived and *surfaced* as [`select::SelectionOutcome::diversity_satisfied`]
//!   rather than buried in a debug log and ignored.
//!
//! It also adds a freshness (seconds-behind) signal the Python version omitted,
//! and a bounded, governance-gated exploration lane (UCB) so unproven indexers
//! can earn a track record without churning the stable set.
#![allow(clippy::doc_lazy_continuation)]

pub mod config;
pub mod domain;
pub mod metrics;
pub mod score;
pub mod select;

pub use config::{ScoringConfig, Weights};
pub use domain::{DeploymentId, DomainError, IndexerId, Utility};
pub use metrics::{
    freshness_utility, latency_utility, price_utility, reliability_ucb, reliability_utility,
    stake_adequacy, uptime_utility, wilson_bounds,
};
pub use score::{score_indexer, AgreementParams, IndexerMetrics, ScoreComponents, Scored};
pub use select::{select_indexers, ChallengeLedger, SelectionOutcome, SelectionRequest};
