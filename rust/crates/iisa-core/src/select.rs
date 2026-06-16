//! The selection engine: add / remove / replace indexers for one deployment.
//!
//! Differences from the Python `IndexerSelector` are all deliberate fixes:
//!
//! * Replacement uses **relative hysteresis + a dwell requirement** carried in a
//!   [`ChallengeLedger`], not a +0.50 absolute wall behind a 0.15 floor
//!   (Finding 3).
//! * Decentralisation is **binding and surfaced**: when it can't be met the
//!   outcome says so (`diversity_satisfied = false`) instead of silently
//!   falling back to the best scorer (Finding 7).
//! * A bounded, governance-gated **exploration lane** (UCB) lets unproven
//!   indexers be tried without the zero-history → 1.0 exploit fill of Finding 5.
//! * The whole function is **pure and order-independent** (Finding 6).

use std::cmp::Ordering;
use std::collections::{BTreeMap, HashSet};

use crate::config::ScoringConfig;
use crate::domain::{DeploymentId, IndexerId};
use crate::score::{score_indexer, AgreementParams, IndexerMetrics, Scored};

/// Per-incumbent record of how many consecutive evaluations a given challenger
/// has cleared the hysteresis threshold. Persist this between daily calls; an
/// empty ledger means "no live challenges".
#[derive(Clone, Debug, Default, PartialEq)]
pub struct ChallengeLedger {
    /// incumbent → (challenger, consecutive valid days)
    pub streaks: BTreeMap<IndexerId, (IndexerId, u32)>,
}

#[derive(Clone, Debug)]
pub struct SelectionRequest {
    pub deployment_id: DeploymentId,
    pub metrics: Vec<IndexerMetrics>,
    pub existing_group: Vec<IndexerId>,
    pub target_size: usize,
    pub denylist: Vec<IndexerId>,
    pub pending: Vec<IndexerId>,
    pub declined: Vec<IndexerId>,
    pub synced: HashSet<IndexerId>,
    pub agreement: AgreementParams,
    pub ledger: ChallengeLedger,
}

#[derive(Clone, Debug)]
pub struct SelectionOutcome {
    pub deployment_id: DeploymentId,
    pub group: Vec<IndexerId>,
    pub added: Vec<IndexerId>,
    pub removed: Vec<IndexerId>,
    pub exploration_picks: Vec<IndexerId>,
    /// Whether the final group meets the ASN/geo diversity floor. Surfaced, not
    /// buried — the caller must be able to see when decentralisation failed.
    pub diversity_satisfied: bool,
    /// Slots left unfilled because no eligible candidate remained.
    pub unfilled_slots: usize,
    /// The updated ledger to persist for the next evaluation.
    pub ledger: ChallengeLedger,
    /// Full per-indexer scores, for transparency/disputability.
    pub scores: Vec<Scored>,
}

/// Run selection for one deployment. Pure: identical inputs → identical output.
pub fn select_indexers(req: SelectionRequest, cfg: &ScoringConfig) -> SelectionOutcome {
    // Score everything, then sort into a stable, deterministic order.
    let mut scored: Vec<Scored> = req
        .metrics
        .iter()
        .map(|m| score_indexer(m, cfg, &req.agreement))
        .collect();
    sort_by_score_then_id(&mut scored);

    let unpickable: HashSet<IndexerId> = req
        .denylist
        .iter()
        .chain(req.pending.iter())
        .chain(req.declined.iter())
        .cloned()
        .collect();

    let mut group: Vec<IndexerId> = req.existing_group.clone();
    let initial: Vec<IndexerId> = group.clone();
    let target = req.target_size;

    let mut explore_picks: Vec<IndexerId> = Vec::new();
    let mut added_this_call: HashSet<IndexerId> = HashSet::new();

    // --- Over-staffed: remove worst, keeping diversity where possible. ---
    if group.len() > target {
        remove_phase(&mut group, &scored, cfg, target);
    }

    // --- Under-staffed: add best, with a reserved exploration lane. ---
    if group.len() < target {
        add_phase(
            &mut group,
            &scored,
            &unpickable,
            &req.synced,
            cfg,
            target,
            &mut explore_picks,
            &mut added_this_call,
        );
    }

    // --- At target: hysteresis + dwell replacement check. ---
    let ledger_out = if group.len() == target {
        replace_phase(
            &mut group,
            &scored,
            &unpickable,
            &req.synced,
            &req.ledger,
            &added_this_call,
            cfg,
        )
    } else {
        ChallengeLedger::default()
    };

    let diversity_satisfied = meets_diversity(&group, &scored, cfg);
    let unfilled_slots = target.saturating_sub(group.len());

    let initial_set: HashSet<&IndexerId> = initial.iter().collect();
    let final_set: HashSet<&IndexerId> = group.iter().collect();
    let added: Vec<IndexerId> = group
        .iter()
        .filter(|i| !initial_set.contains(*i))
        .cloned()
        .collect();
    let removed: Vec<IndexerId> = initial
        .iter()
        .filter(|i| !final_set.contains(*i))
        .cloned()
        .collect();

    SelectionOutcome {
        deployment_id: req.deployment_id,
        group,
        added,
        removed,
        exploration_picks: explore_picks,
        diversity_satisfied,
        unfilled_slots,
        ledger: ledger_out,
        scores: scored,
    }
}

// --------------------------------------------------------------------------
// Phases
// --------------------------------------------------------------------------

#[allow(clippy::too_many_arguments)]
fn add_phase(
    group: &mut Vec<IndexerId>,
    scored: &[Scored],
    unpickable: &HashSet<IndexerId>,
    synced: &HashSet<IndexerId>,
    cfg: &ScoringConfig,
    target: usize,
    explore_picks: &mut Vec<IndexerId>,
    added_this_call: &mut HashSet<IndexerId>,
) {
    let to_fill = target - group.len();
    let explore_slots = ((target as f64) * cfg.exploration_fraction).round() as usize;
    let explore_slots = explore_slots.min(to_fill);
    let exploit_fill = to_fill - explore_slots;

    for slot in 0..to_fill {
        let cands = eligible(scored, group, unpickable);
        if cands.is_empty() {
            break;
        }
        let use_explore = slot >= exploit_fill;
        let pick = if use_explore {
            pick_explore(&cands, cfg)
        } else {
            pick_exploit(&cands, synced)
        };
        match pick {
            Some(id) => {
                if use_explore {
                    explore_picks.push(id.clone());
                }
                added_this_call.insert(id.clone());
                group.push(id);
            }
            None => break,
        }
    }

    // Binding diversity repair (Finding 7): try to reach the ASN/geo floor by
    // swapping in candidates that add a missing org/region. If impossible, the
    // group is left as-is and `diversity_satisfied` will report false.
    if cfg.diversity_binding {
        enforce_diversity(group, scored, unpickable, cfg);
    }
}

fn remove_phase(group: &mut Vec<IndexerId>, scored: &[Scored], cfg: &ScoringConfig, target: usize) {
    while group.len() > target {
        // Worst-scoring first.
        let mut members = group.clone();
        members.sort_by(|a, b| {
            score_of(scored, a)
                .partial_cmp(&score_of(scored, b))
                .unwrap_or(Ordering::Equal)
                .then(a.cmp(b))
        });

        let mut removed_one = false;
        for m in &members {
            let trial: Vec<IndexerId> = group.iter().filter(|i| *i != m).cloned().collect();
            let keeps_diversity =
                !cfg.diversity_binding || trial.len() < 2 || meets_diversity(&trial, scored, cfg);
            if keeps_diversity {
                group.retain(|i| i != m);
                removed_one = true;
                break;
            }
        }
        if !removed_one {
            // Can't preserve diversity; drop the absolute worst and stop forcing.
            if let Some(worst) = members.first() {
                group.retain(|i| i != worst);
            } else {
                break;
            }
        }
    }
}

#[allow(clippy::too_many_arguments)]
fn replace_phase(
    group: &mut Vec<IndexerId>,
    scored: &[Scored],
    unpickable: &HashSet<IndexerId>,
    synced: &HashSet<IndexerId>,
    prev_ledger: &ChallengeLedger,
    added_this_call: &HashSet<IndexerId>,
    cfg: &ScoringConfig,
) -> ChallengeLedger {
    let mut new_ledger = ChallengeLedger::default();

    // Process incumbents weakest-first for determinism.
    let mut incumbents = group.clone();
    incumbents.sort_by(|a, b| {
        score_of(scored, a)
            .partial_cmp(&score_of(scored, b))
            .unwrap_or(Ordering::Equal)
            .then(a.cmp(b))
    });

    for inc in incumbents {
        // Never challenge an indexer we added in this same call.
        if added_this_call.contains(&inc) {
            continue;
        }
        if !group.contains(&inc) {
            continue; // already swapped out this call
        }
        let inc_score = score_of(scored, &inc);

        let cands = eligible(scored, group, unpickable);
        let Some(challenger) = pick_exploit_ref(&cands, synced) else {
            continue;
        };

        // Hysteresis: must beat the incumbent by (1 + delta), plus the
        // amortised sync-cost penalty. No absolute +0.50 wall.
        let threshold = (1.0 + cfg.replacement_delta) * inc_score + cfg.sync_cost;
        let valid = challenger.score > inc_score && challenger.score >= threshold;
        if !valid {
            continue; // streak (if any) lapses — not carried forward
        }

        // The swap must not break the diversity floor.
        let mut trial: Vec<IndexerId> = group.iter().filter(|i| **i != inc).cloned().collect();
        trial.push(challenger.indexer.clone());
        if cfg.diversity_binding && !meets_diversity(&trial, scored, cfg) {
            continue;
        }

        // Dwell: the same challenger must clear the bar on consecutive days.
        let streak = match prev_ledger.streaks.get(&inc) {
            Some((prev_ch, n)) if *prev_ch == challenger.indexer => n + 1,
            _ => 1,
        };
        if streak >= cfg.dwell_days {
            *group = trial; // execute the replacement
        } else {
            new_ledger
                .streaks
                .insert(inc.clone(), (challenger.indexer.clone(), streak));
        }
    }

    new_ledger
}

// --------------------------------------------------------------------------
// Helpers
// --------------------------------------------------------------------------

fn sort_by_score_then_id(scored: &mut [Scored]) {
    scored.sort_by(|a, b| {
        b.score
            .partial_cmp(&a.score)
            .unwrap_or(Ordering::Equal)
            .then(a.indexer.cmp(&b.indexer))
    });
}

fn eligible<'a>(
    scored: &'a [Scored],
    group: &[IndexerId],
    unpickable: &HashSet<IndexerId>,
) -> Vec<&'a Scored> {
    scored
        .iter()
        .filter(|s| {
            s.eligible && !group.iter().any(|g| g == &s.indexer) && !unpickable.contains(&s.indexer)
        })
        .collect()
}

/// Best by exploit score; among near-equals prefer already-synced candidates
/// (so queries serve right after acceptance), then break ties by id.
fn pick_exploit(cands: &[&Scored], synced: &HashSet<IndexerId>) -> Option<IndexerId> {
    pick_exploit_ref(cands, synced).map(|s| s.indexer.clone())
}

fn pick_exploit_ref<'a>(cands: &[&'a Scored], synced: &HashSet<IndexerId>) -> Option<&'a Scored> {
    cands.iter().copied().max_by(|a, b| {
        a.score
            .partial_cmp(&b.score)
            .unwrap_or(Ordering::Equal)
            .then(
                synced
                    .contains(&a.indexer)
                    .cmp(&synced.contains(&b.indexer)),
            )
            // For a stable max, the *smaller* id should win ties, so invert.
            .then(b.indexer.cmp(&a.indexer))
    })
}

/// Exploration pick: highest reliability UCB among unproven candidates.
fn pick_explore(cands: &[&Scored], cfg: &ScoringConfig) -> Option<IndexerId> {
    let mut pool: Vec<&Scored> = cands
        .iter()
        .copied()
        .filter(|s| s.trials < cfg.proven_trials)
        .collect();
    if pool.is_empty() {
        pool = cands.to_vec();
    }
    pool.into_iter()
        .max_by(|a, b| {
            a.reliability_ucb
                .partial_cmp(&b.reliability_ucb)
                .unwrap_or(Ordering::Equal)
                .then(b.indexer.cmp(&a.indexer))
        })
        .map(|s| s.indexer.clone())
}

fn score_of(scored: &[Scored], id: &IndexerId) -> f64 {
    scored
        .iter()
        .find(|s| &s.indexer == id)
        .map(|s| s.score)
        .unwrap_or(0.0)
}

fn region_key(geo: Option<(f64, f64)>, deg: f64) -> Option<(i64, i64)> {
    let d = if deg > 0.0 { deg } else { 1.0 };
    geo.map(|(la, lo)| ((la / d).floor() as i64, (lo / d).floor() as i64))
}

fn diversity_sets(
    group: &[IndexerId],
    scored: &[Scored],
    cfg: &ScoringConfig,
) -> (HashSet<String>, HashSet<(i64, i64)>) {
    let mut orgs = HashSet::new();
    let mut regions = HashSet::new();
    for id in group {
        if let Some(s) = scored.iter().find(|s| &s.indexer == id) {
            if let Some(o) = &s.org {
                orgs.insert(o.clone());
            }
            if let Some(r) = region_key(s.geo, cfg.geo_region_degrees) {
                regions.insert(r);
            }
        }
    }
    (orgs, regions)
}

/// A group of fewer than 2 trivially passes; otherwise it must clear both the
/// distinct-orgs and distinct-regions floors.
fn meets_diversity(group: &[IndexerId], scored: &[Scored], cfg: &ScoringConfig) -> bool {
    if group.len() < 2 {
        return true;
    }
    let (orgs, regions) = diversity_sets(group, scored, cfg);
    orgs.len() >= cfg.min_distinct_orgs && regions.len() >= cfg.min_distinct_regions
}

/// Try to reach the diversity floor by swapping the lowest-scoring member for
/// the best candidate that adds a missing org/region. Returns once satisfied or
/// no further progress is possible.
fn enforce_diversity(
    group: &mut Vec<IndexerId>,
    scored: &[Scored],
    unpickable: &HashSet<IndexerId>,
    cfg: &ScoringConfig,
) {
    let max_iters = group.len() + 8;
    for _ in 0..max_iters {
        if meets_diversity(group, scored, cfg) {
            return;
        }
        let (orgs, regions) = diversity_sets(group, scored, cfg);

        // Best-scoring eligible candidate that adds diversity.
        let mut helpers: Vec<&Scored> = eligible(scored, group, unpickable)
            .into_iter()
            .filter(|c| {
                let adds_org = c.org.as_ref().is_some_and(|o| !orgs.contains(o));
                let adds_region = region_key(c.geo, cfg.geo_region_degrees)
                    .is_some_and(|r| !regions.contains(&r));
                adds_org || adds_region
            })
            .collect();
        sort_refs_by_score_then_id(&mut helpers);
        let Some(helper) = helpers.first().map(|s| s.indexer.clone()) else {
            return; // nothing can improve diversity
        };

        // Remove the current lowest-scoring member.
        let worst = group
            .iter()
            .min_by(|a, b| {
                score_of(scored, a)
                    .partial_cmp(&score_of(scored, b))
                    .unwrap_or(Ordering::Equal)
                    .then(a.cmp(b))
            })
            .cloned();
        let Some(worst) = worst else {
            return;
        };
        if worst == helper {
            return; // no progress possible
        }
        group.retain(|i| i != &worst);
        group.push(helper);
    }
}

fn sort_refs_by_score_then_id(refs: &mut [&Scored]) {
    refs.sort_by(|a, b| {
        b.score
            .partial_cmp(&a.score)
            .unwrap_or(Ordering::Equal)
            .then(a.indexer.cmp(&b.indexer))
    });
}
