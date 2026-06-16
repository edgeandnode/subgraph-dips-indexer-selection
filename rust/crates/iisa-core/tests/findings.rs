//! One proving test per finding in IISA-AUDIT-FINDINGS.md, plus the
//! price-ceiling correction. Each demonstrates that the Rust rewrite does NOT
//! exhibit the Python behaviour the audit flagged.

use std::collections::HashSet;

use iisa_core::*;

/// A "good, busy, well-staked" indexer as a baseline; tests mutate fields.
fn good(id: &str) -> IndexerMetrics {
    let mut m = IndexerMetrics::new(IndexerId::new(id));
    m.query_successes = 99_000.0;
    m.query_trials = 100_000.0;
    m.uptime_ratio = 0.999;
    m.uptime_observed_secs = 2_000_000.0;
    m.latency_ci_upper = 100.0;
    m.seconds_behind = 0.0;
    m.slashable_stake_grt = 1_000_000.0;
    m.price_per_30d_grt = Some(50.0);
    m.asn_org = Some(format!("org-{id}"));
    m.geo_lat = Some(10.0);
    m.geo_lon = Some(10.0);
    m
}

fn agreement() -> AgreementParams {
    AgreementParams {
        price_ceiling: 100.0,
        value_at_risk: 100_000.0,
    }
}

fn score(m: &IndexerMetrics) -> Scored {
    score_indexer(m, &ScoringConfig::default(), &agreement())
}

// ---------------------------------------------------------------------------
// Finding 1 — stake adequacy is not inverted, and saturates.
// ---------------------------------------------------------------------------
#[test]
fn finding1_stake_is_not_divided_by_fees() {
    // An idle indexer with enormous stake but a tiny track record must NOT beat
    // a busy, adequately-staked one. (Old code: idle/zero-fee → 1.0 on 30%.)
    let mut idle = good("idle");
    idle.query_successes = 5.0;
    idle.query_trials = 5.0;
    idle.slashable_stake_grt = 1_000_000_000.0; // huge

    let busy = good("busy"); // 99k/100k successes, adequate stake

    let idle_s = score(&idle);
    let busy_s = score(&busy);

    assert!(idle_s.eligible && busy_s.eligible);
    assert!(
        busy_s.score > idle_s.score,
        "commercial activity must not be penalised: busy={} idle={}",
        busy_s.score,
        idle_s.score
    );

    // Stake adequacy saturates: 10x more stake past the cap adds nothing.
    let a = stake_adequacy(1_000_000_000.0, 100_000.0, 2.0).get();
    let b = stake_adequacy(1_000_000_000_000.0, 100_000.0, 2.0).get();
    assert_eq!(a, 1.0);
    assert_eq!(b, 1.0);
}

// ---------------------------------------------------------------------------
// Finding 2 — quality is absolute and confidence-aware, not 0.97-of-best.
// ---------------------------------------------------------------------------
#[test]
fn finding2_quality_is_absolute_not_field_relative() {
    let cfg = ScoringConfig::default();

    // A 96.9%-uptime indexer is judged on its own merit, regardless of any
    // 99.99% outlier elsewhere in the field (utilities never reference others).
    let mid = uptime_utility(0.969, 5_000_000.0, &cfg).get();
    assert!(mid > 0.0, "an acceptable indexer is not floored to 0");

    // Confidence is rewarded: 100% over 5 trials must score below 99.9% over 1M.
    let tiny_sample = reliability_utility(5.0, 5.0, &cfg).get();
    let huge_sample = reliability_utility(999_000.0, 1_000_000.0, &cfg).get();
    assert!(
        huge_sample > tiny_sample,
        "confidence must beat a small lucky sample: 1M={huge_sample} 5={tiny_sample}"
    );
}

// ---------------------------------------------------------------------------
// Finding 3 — incumbency uses hysteresis + dwell, not a +0.50 wall.
// ---------------------------------------------------------------------------
#[test]
fn finding3_replacement_requires_sustained_dwell() {
    // delta 0.10, dwell 3, sync_cost 0. Relax diversity so this test isolates
    // the hysteresis/dwell behaviour from the (separately tested) Finding 7.
    let cfg = ScoringConfig {
        min_distinct_orgs: 1,
        min_distinct_regions: 1,
        ..Default::default()
    };

    // Weak incumbent (poor latency drags its score down) + two solid ones.
    let mut weak = good("weak");
    weak.latency_ci_upper = 5_000.0;
    let g1 = good("g1");
    let g2 = good("g2");
    let challenger = good("challenger"); // strong, well clear of weak

    let metrics = vec![weak.clone(), g1, g2, challenger];
    let group = vec![
        IndexerId::new("weak"),
        IndexerId::new("g1"),
        IndexerId::new("g2"),
    ];

    let run = |ledger: ChallengeLedger| {
        select_indexers(
            SelectionRequest {
                deployment_id: DeploymentId::new("Qm1"),
                metrics: metrics.clone(),
                existing_group: group.clone(),
                target_size: 3,
                denylist: vec![],
                pending: vec![],
                declined: vec![],
                synced: HashSet::new(),
                agreement: agreement(),
                ledger,
            },
            &cfg,
        )
    };

    // Day 1 & 2: challenge accrues but no swap yet.
    let d1 = run(ChallengeLedger::default());
    assert!(d1.removed.is_empty(), "no replacement on day 1");
    assert!(d1.group.contains(&IndexerId::new("weak")));

    let d2 = run(d1.ledger);
    assert!(d2.removed.is_empty(), "no replacement on day 2");

    // Day 3: dwell satisfied → the weak incumbent is finally replaced.
    let d3 = run(d2.ledger);
    assert_eq!(d3.removed, vec![IndexerId::new("weak")]);
    assert!(d3.group.contains(&IndexerId::new("challenger")));
}

#[test]
fn finding3_marginal_challenger_never_replaces() {
    let cfg = ScoringConfig {
        min_distinct_orgs: 1,
        min_distinct_regions: 1,
        ..Default::default()
    };

    // Incumbent and challenger differ only slightly on price (< 10% better),
    // so the challenger can never clear the (1 + delta) hysteresis bar.
    let mut incumbent = good("inc");
    incumbent.price_per_30d_grt = Some(60.0);
    let mut challenger = good("chal");
    challenger.price_per_30d_grt = Some(55.0);
    let g1 = good("g1");
    let g2 = good("g2");

    let metrics = vec![incumbent, challenger, g1, g2];
    let group = vec![
        IndexerId::new("inc"),
        IndexerId::new("g1"),
        IndexerId::new("g2"),
    ];

    let mut ledger = ChallengeLedger::default();
    for day in 0..10 {
        let out = select_indexers(
            SelectionRequest {
                deployment_id: DeploymentId::new("Qm1"),
                metrics: metrics.clone(),
                existing_group: group.clone(),
                target_size: 3,
                denylist: vec![],
                pending: vec![],
                declined: vec![],
                synced: HashSet::new(),
                agreement: agreement(),
                ledger,
            },
            &cfg,
        );
        assert!(
            out.removed.is_empty(),
            "marginal challenger must never replace (day {day})"
        );
        ledger = out.ledger;
    }
}

// ---------------------------------------------------------------------------
// Finding 4 — weighted product vetoes; capital can't buy off bad quality.
// ---------------------------------------------------------------------------
#[test]
fn finding4_near_zero_quality_vetoes_the_score() {
    // Perfect price and stake, but zero successful queries over a large sample.
    let mut broken = good("broken");
    broken.query_successes = 0.0;
    broken.query_trials = 100_000.0;
    broken.price_per_30d_grt = Some(0.0); // cheapest possible
    broken.slashable_stake_grt = 1_000_000_000.0;

    let s = score(&broken);
    // The product crushes it toward zero — capital cannot buy it back.
    assert!(
        s.score < 0.05,
        "near-zero reliability must crush the score regardless of price/stake: {}",
        s.score
    );
    // And it loses decisively (>10x) to a balanced indexer.
    let balanced = score(&good("ok")).score;
    assert!(
        balanced > 10.0 * s.score,
        "balanced ({balanced}) must dominate broken ({})",
        s.score
    );
}

// ---------------------------------------------------------------------------
// Finding 5 — no-history fills DOWN (conservative), and only the explore lane
// can still try it via UCB. (Old code filled zero-fee UP to 1.0.)
// ---------------------------------------------------------------------------
#[test]
fn finding5_no_history_is_conservative_not_inflated() {
    let cfg = ScoringConfig::default();

    let mut fresh = good("fresh");
    fresh.query_successes = 0.0;
    fresh.query_trials = 0.0;
    fresh.uptime_observed_secs = 0.0;
    fresh.slashable_stake_grt = 1_000_000_000.0; // huge stake

    let s = score_indexer(&fresh, &cfg, &agreement());
    // Conservative (low), and crucially NOT inflated to ~1.0 the way the old
    // zero-fee → 1.0 fill would have done despite the huge stake.
    assert!(
        s.score < 0.4,
        "no track record must yield a conservative score, not inflated: {}",
        s.score
    );
    assert!(
        s.score < score(&good("ok")).score,
        "a fresh indexer must not out-score a proven one on the exploit axis"
    );

    // But the exploration lane still sees promise (optimistic upper bound).
    assert!(
        s.reliability_ucb > 0.9,
        "exploration lane should still find an unproven indexer worth trying: {}",
        s.reliability_ucb
    );

    // With the lane enabled, a fresh indexer can actually be picked.
    let mut explore_cfg = cfg.clone();
    explore_cfg.exploration_fraction = 0.34; // ~1 of 3 slots
    explore_cfg.min_distinct_orgs = 1; // isolate the exploration behaviour
    explore_cfg.min_distinct_regions = 1;

    let metrics = vec![good("a"), good("b"), fresh.clone()];
    let out = select_indexers(
        SelectionRequest {
            deployment_id: DeploymentId::new("Qm1"),
            metrics,
            existing_group: vec![],
            target_size: 3,
            denylist: vec![],
            pending: vec![],
            declined: vec![],
            synced: HashSet::new(),
            agreement: agreement(),
            ledger: ChallengeLedger::default(),
        },
        &explore_cfg,
    );
    assert!(
        out.exploration_picks.contains(&IndexerId::new("fresh")),
        "exploration lane should pick the unproven indexer, got {:?}",
        out.exploration_picks
    );
}

// ---------------------------------------------------------------------------
// Finding 6 — selection is deterministic and order-independent.
// ---------------------------------------------------------------------------
#[test]
fn finding6_selection_is_deterministic() {
    let cfg = ScoringConfig::default();
    let base = vec![
        loc(good("a"), 10.0, 10.0, "org-a"),
        loc(good("b"), 40.0, 40.0, "org-b"),
        loc(good("c"), 70.0, 70.0, "org-c"),
        loc(good("d"), 20.0, 80.0, "org-d"),
    ];

    let make = |metrics: Vec<IndexerMetrics>| SelectionRequest {
        deployment_id: DeploymentId::new("Qm1"),
        metrics,
        existing_group: vec![],
        target_size: 3,
        denylist: vec![],
        pending: vec![],
        declined: vec![],
        synced: HashSet::new(),
        agreement: agreement(),
        ledger: ChallengeLedger::default(),
    };

    let out1 = select_indexers(make(base.clone()), &cfg);
    let out2 = select_indexers(make(base.clone()), &cfg);
    assert_eq!(
        out1.group, out2.group,
        "identical inputs → identical output"
    );

    // Reversed input order must not change the result.
    let mut reversed = base.clone();
    reversed.reverse();
    let out3 = select_indexers(make(reversed), &cfg);
    assert_eq!(out1.group, out3.group, "result is input-order independent");
}

// ---------------------------------------------------------------------------
// Finding 7 — diversity is ASN/geo-based and binding (surfaced, not bypassed).
// ---------------------------------------------------------------------------
#[test]
fn finding7_diversity_is_binding_and_surfaced() {
    let cfg = ScoringConfig::default(); // needs >=2 orgs and >=2 regions

    // Case A: all candidates share one org and one region → unsatisfiable.
    let same = vec![
        loc(good("a"), 10.0, 10.0, "same-org"),
        loc(good("b"), 11.0, 11.0, "same-org"),
        loc(good("c"), 12.0, 12.0, "same-org"),
    ];
    let out_a = select_indexers(
        SelectionRequest {
            deployment_id: DeploymentId::new("Qm1"),
            metrics: same,
            existing_group: vec![],
            target_size: 3,
            denylist: vec![],
            pending: vec![],
            declined: vec![],
            synced: HashSet::new(),
            agreement: agreement(),
            ledger: ChallengeLedger::default(),
        },
        &cfg,
    );
    assert!(
        !out_a.diversity_satisfied,
        "an undiversifiable field must be reported, not silently accepted"
    );

    // Case B: distinct orgs and regions → satisfiable.
    let diverse = vec![
        loc(good("a"), 10.0, 10.0, "org-a"),
        loc(good("b"), 40.0, 40.0, "org-b"),
        loc(good("c"), 70.0, 70.0, "org-c"),
    ];
    let out_b = select_indexers(
        SelectionRequest {
            deployment_id: DeploymentId::new("Qm1"),
            metrics: diverse,
            existing_group: vec![],
            target_size: 3,
            denylist: vec![],
            pending: vec![],
            declined: vec![],
            synced: HashSet::new(),
            agreement: agreement(),
            ledger: ChallengeLedger::default(),
        },
        &cfg,
    );
    assert!(out_b.diversity_satisfied);
    assert_eq!(out_b.group.len(), 3);
}

// ---------------------------------------------------------------------------
// Recommendation 3 — price is anchored to the ceiling, not the observed max.
// ---------------------------------------------------------------------------
#[test]
fn price_is_anchored_to_ceiling_not_observed_max() {
    // 50 against a ceiling of 100 with a 0 floor is always 0.5, no matter what
    // anyone else bids — a lone high bidder cannot compress the axis.
    assert_eq!(price_utility(50.0, 100.0, 0.0).get(), 0.5);

    // Above the ceiling → ineligible (gated out), not merely low-scoring.
    let mut pricey = good("pricey");
    pricey.price_per_30d_grt = Some(150.0);
    let s = score(&pricey);
    assert!(!s.eligible);
    assert_eq!(s.score, 0.0);
}

// --- helpers ---
fn loc(mut m: IndexerMetrics, lat: f64, lon: f64, org: &str) -> IndexerMetrics {
    m.geo_lat = Some(lat);
    m.geo_lon = Some(lon);
    m.asn_org = Some(org.to_string());
    m
}
