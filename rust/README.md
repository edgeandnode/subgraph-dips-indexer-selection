# IISA selection core (Rust)

A dependency-free Rust rewrite of the IISA **selection algorithm** — the part of
`src/iisa/indexer_selection.py` that every finding in
[`../IISA-AUDIT-FINDINGS.md`](../IISA-AUDIT-FINDINGS.md) was filed against.

The surrounding plumbing (Kafka replay, GeoIP, the FastAPI service) is unchanged
and stays in Python; this crate is the corrected *brain* it can call. No finding
was filed against the plumbing, so it wasn't ported.

## What changed, and where it's enforced

| Finding | Old (Python) | New (this crate) |
| --- | --- | --- |
| 1 — inverted, global `stake_to_fees` | `stake / fees`, zero-fee → 1.0, weight 0.30 | `metrics::stake_adequacy`: bounded `stake / (κ·value_at_risk)`, saturating, weight 0.05 |
| 2 — floating 0.97-of-best quality gate | relative to field best, success @ 0.05 | `metrics::reliability_utility` / `uptime_utility`: Wilson LCB vs an **absolute** SLA, reliability is top-weighted |
| 3 — +0.50 incumbency wall | absolute margin behind 0.15 floor | `select`: relative hysteresis `(1+δ)` + a dwell streak (`ChallengeLedger`) |
| 4 — additive sum buys off quality | weighted average | `score::weighted_product`: weighted geometric mean → near-zero quality vetoes |
| 5 — asymmetric NaN fill | fees fill up, quality fills down | Beta prior + LCB everywhere; nothing fills to 1.0 |
| 6 — determinism | seeded-but-biased | `select_indexers` is pure and input-order independent |
| 7 — org-counting, silently bypassed | binary ≥2 check, then ignored | ASN/geo diversity, **surfaced** as `diversity_satisfied` |

Plus two additions: a **freshness** (seconds-behind) signal the Python omitted,
and a bounded, governance-gated **UCB exploration lane** so unproven indexers can
earn a record without churning the stable set.

The `Utility` newtype makes the headline bugs *uncompilable*: it can only hold a
value in `[0, 1]`, so "zero fees → fill to 1.0" has nowhere to live.

## Layout

```
crates/iisa-core/
  src/domain.rs   IndexerId, DeploymentId, Utility (illegal states unrepresentable)
  src/config.rs   ScoringConfig — every governance knob, with justified defaults
  src/metrics.rs  Wilson bounds + the absolute per-metric utility transforms
  src/score.rs    IndexerMetrics → weighted-product score, behind hard gates
  src/select.rs   add / remove / replace with hysteresis, dwell, diversity, exploration
  tests/findings.rs  one proving test per finding
```

## Build & test

```bash
cd rust
cargo test      # 9/9 proving tests
cargo clippy --all-targets
```

## Note on the wire format

Reliability is now a confidence bound, so the scoring job must emit
`query_successes` / `query_trials` (counts), not just a success ratio, and a
`seconds_behind` value for freshness. See `score::IndexerMetrics`.
