# IISA Audit — Verified Findings

A teardown of the DIPs Indexing Indexer Selection Algorithm was checked line-by-line
against the source in this repo. This document keeps **only what the code actually
confirms** — overstated, mis-described, and incorrect claims from the original critique
have been dropped or corrected. Every finding below cites the source.

Scope read: `src/iisa/indexer_selection.py`, `src/iisa/score_columns.py`,
`cronjobs/compute_scores/processing.py`, `cronjobs/compute_scores/redpanda.py`.

---

## Summary

The selector combines six min-max-normalised metrics into a **weighted sum** and
assigns/replaces indexers under an incumbency rule, a quality gate, and a
decentralisation floor. Four structural problems are confirmed and load-bearing:

1. The dominant signal (`stake_to_fees`, 0.30) is inverted and computed **globally**,
   so commercial success is penalised on every deployment.
2. Quality is a *floating relative* gate worth only 0.20 combined, with data
   correctness (success rate) at 0.05.
3. Incumbency is near-unbreakable (+0.50 margin on a 0–1 scale, behind a 0.15 floor).
4. Scoring is an additive weighted sum, so capital can compensate for poor quality
   instead of a near-zero quality vetoing the candidate.

Two further issues are real but narrower: degraded mode fails open onto price, and the
decentralisation check is a binary floor (though it is ASN/GeoIP-derived, **not**
trivially spoofable self-reported labels).

---

## Finding 1 — Economic-security signal is inverted and global

**Severity: High. Confirmed, aggravated.**

`stake_to_fees = slashable_stake / total_query_fees`. Holding stake constant, *more*
fees earned produces a strictly *lower* score. Indexers with zero fees divide by zero →
NaN, then fill **above** the max finite value, normalising to **1.0** — so a high-stake
address that has never served a query maxes out the highest-weighted axis.

```python
# cronjobs/compute_scores/redpanda.py:614
df["stake_to_fees"] = df["last_known_slashable_stake"] / df["total_query_fees"].replace(
    0.0, float("nan")
)
```
```python
# src/iisa/indexer_selection.py:659-665
stf = merged["stake_to_fees"].copy()
finite_max = stf.max()
fill_value = (finite_max + 1.0) if pd.notna(finite_max) else 1.0
stf = stf.fillna(fill_value)
merged["norm_stake_to_fees"] = _normalize_generic(stf)
```

**Computed globally, not per-deployment.** `_fees_per_indexer` is keyed by indexer alone
and summed across the entire replay window over *all* deployments, then merged on
`indexer` (`redpanda.py:481, 610`; `processing.py:1531`). The same fee-penalised score is
applied to every deployment the indexer bids on — fees earned elsewhere drag it down
everywhere.

**Partial mitigation already present (do not ignore):** `optimistic_dips_fees` adds
dipper's expected DIPs fees to incumbents' query fees before recomputing the ratio, a
deliberate load-balancing counter (`indexer_selection.py:177-201`). It does **not** close
the cold-start hole — a never-seen address still fills to 1.0.

**Fix direction:** replace with a bounded stake-adequacy ratio
`A = min(1, slashable_stake / (κ · value_at_risk))` tied to the agreement's slashable
exposure, not historical revenue. Saturating, not inverted.

---

## Finding 2 — Quality is a floating relative gate, and under-weighted

**Severity: High. Confirmed exactly.**

Uptime and success rate are scored against `best * 0.97` — i.e. 97% **of the best
indexer in the field**, not an absolute SLA. Anything below the bar floors to 0; from the
bar to the best it scales linearly to 1.

```python
# src/iisa/indexer_selection.py:34, 747-758
NON_ZERO_UPTIME_SUCCESS_RATE_SCORE_THRESHOLD = 0.97
...
best = series.max()
threshold = best * NON_ZERO_UPTIME_SUCCESS_RATE_SCORE_THRESHOLD
normalized = series.apply(lambda x: max(0, min(1, (x - threshold) / (best - threshold))))
```

Consequences:
- **The bar floats.** One indexer at 99.99% pushes the threshold to ~96.99%; a
  perfectly acceptable 96.9%-uptime indexer scores 0 purely because the field has an
  outlier.
- **It mostly subtracts.** No smooth reward for excellent vs. adequate.
- **Correctness is priced at 0.05.** `success_rate` — the network's entire value
  proposition — carries one-sixth the weight of price.
- **Unstable on small sets.** With two candidates the metric is near-binary.

**Fix direction:** score reliability/uptime with a Wilson/Beta-Bernoulli lower bound
against an **absolute** target (e.g. 99.5%), and raise combined quality weight well above
0.20.

---

## Finding 3 — Incumbency is near-unbreakable

**Severity: High. Confirmed exactly.**

A challenger replaces an incumbent only if **both** hold:
- the incumbent scores **below 0.15** (`MIN_INDEXER_SCORE`), and
- the challenger scores **at least incumbent + 0.50** (`REPLACEMENT_MARGIN`), on a 0–1
  scale.

```python
# src/iisa/indexer_selection.py:38, 42, 450, 477
MIN_INDEXER_SCORE = 0.15
REPLACEMENT_MARGIN = 0.50
...
if current_score >= MIN_INDEXER_SCORE:   # not a replacement candidate
    continue
...
if candidate_score > current_score + REPLACEMENT_MARGIN:   # must beat by +0.50
```

A 0.50 margin is half the entire dynamic range. Combined with Finding 1 (an incumbent
that earns fees sees its dominant axis decay over time, yet is protected until it falls
under 0.15), early winners are structurally entrenched. The cheapest route past the wall
is to have arrived first, not to be better.

**Fix direction:** relative hysteresis plus dwell time — e.g. replace only if
`S_challenger ≥ (1+δ)·S_incumbent` (δ ≈ 0.10) sustained over D consecutive daily
evaluations — and amortise the displacement test by the unrecovered initial-sync cost so
churn is discouraged when re-sync is expensive and cheap when it isn't.

---

## Finding 4 — Additive model lets capital buy off poor quality

**Severity: High. Confirmed.**

The final score is a weighted **average** (sum over present metrics, renormalised by the
weights actually present), not a product:

```python
# src/iisa/indexer_selection.py:786-796
weighted_sum = np.nansum(matrix * weight_vector, axis=1)
weight_total = present @ weight_vector
return pd.Series(weighted_sum / weight_total, index=df.index)
```

A near-zero success rate costs at most 0.05 of the total and is fully compensable by a
strong price or stake score. A correctness failure should **veto** a candidate.

**Fix direction:** switch to a weighted **product** `S = Π_i u_iʷⁱ` (aligning with the
gateway's battle-tested `candidate-selection` crate), so a near-zero quality utility
drives the whole score toward zero.

---

## Finding 5 — NaN-fill direction is asymmetric

**Severity: Medium. Confirmed.**

Missing data is treated inconsistently across axes:
- **Zero/missing fees → fills UP** to 1.0 on economic security
  (`indexer_selection.py:663`).
- **Missing quality → fills DOWN** to 0 (`indexer_selection.py:716`, after the
  uptime/success gate already floors sub-threshold values to 0).

The asymmetry rewards having *no* fee history while punishing having *no* quality
history — exactly backwards for a QoS-paying buyer.

**Fix direction:** treat absence of evidence conservatively and uniformly (confidence
intervals that widen toward a neutral prior, not toward the maximum).

---

## Finding 6 — Degraded mode fails open onto price

**Severity: Medium. Confirmed (corrected description).**

In **full** pipeline failure, `compute_degraded_scores` flattens *all* quality metrics —
including economic security — to 0.5, keeping only real `/dips/info` pricing:

```python
# cronjobs/compute_scores/processing.py:1561-1569
scores["lat_normalized_score"] = 0.5
scores["uptime_score"] = 0.5
scores["success_rate"] = 0.5
scores["norm_uptime_score"] = 0.5
scores["norm_success_rate"] = 0.5
scores["norm_stake_to_fees"] = 0.5
```

So selection collapses onto **price alone** — not "price + inverted econ." (The
"high-stake idle wins" scenario applies only to *partial*, geo-lookup-only degradation,
where latency goes neutral but econ stays live.) Either way it is a silent fail-open with
no "going dark" signal to operators.

**Fix direction:** make degraded mode explicit and observable; consider holding the prior
selection rather than re-ranking on price under measurement failure.

---

## Finding 7 — Decentralisation is a binary floor, and bypassable

**Severity: Medium. Confirmed in weakened form — original "spoofable labels" claim is
FALSE.**

The check requires the resulting group to keep ≥2 unique `org` values and ≥2 unique
`destination_loc` values; groups smaller than 2 always pass
(`indexer_selection.py:331-367`).

**Correction:** `org` is **not** self-reported. It is the **ASN organisation** from
MaxMind GeoLite2-ASN, and location is from GeoLite2-City — both derived from the
indexer's URL → DNS → IP → GeoIP lookup (`processing.py:21-22, 46-47, 143, 752`). The
"label two addresses with two org strings" attack does not work; you would need serving
IPs in two distinct ASNs and two cities.

What remains genuinely valid:
- It is a **binary pass/fail**, not a diversity *score*.
- Renting VMs in two ASNs / cities is cheap, so it is not real sybil resistance.
- It is **not even a hard floor**: when no candidate satisfies it, the selector falls
  back to the best scorer regardless (`indexer_selection.py:593-605`), so the check is
  bypassed by exhaustion.

**Fix direction:** replace the binary check with a group-level diversity *multiplier*
combining ASN/network-prefix spread, stake-cluster distinctness, and (optionally)
behavioural correlation — and make the floor actually binding.

---

## Confirmed-correct, not a defect

- **Price normalisation already supports a ceiling anchor.** When `price_ceiling` is
  supplied, `base_price_per_epoch` is scored as `1 - price/ceiling` (an absolute anchor),
  not against the observed max (`indexer_selection.py:681-694`). The earlier assumption
  that price uses observed-max is **wrong** for the base price. Note `price_per_entity`
  still uses the observed max (`indexer_selection.py:701-710`).

---

## Recommendation summary (validated subset)

| Stage | Action | Status in code |
| --- | --- | --- |
| 0 | Replace inverted `stake_to_fees` with bounded stake-adequacy ratio | Not started |
| 0 | Cut the 0.30 econ weight | Not started |
| 0 | Score price against ceiling, not observed max | **Done for base price**; entity price still observed-max |
| 1 | Weighted **product** instead of sum | Not started |
| 1 | Reliability/uptime via confidence-interval LCB vs. absolute SLA | Not started |
| 2 | Relative hysteresis + dwell time instead of +0.50 margin | Not started |
| 3 | Thompson-sampling exploration lane for unproven indexers | Not started |
| 4 | Diversity *score* (ASN already used) instead of binary ≥2 floor | Partially — ASN/GeoIP inputs exist, scoring does not |
| 5 | Publish algorithm + parameters under governance; log decisions; allow dispute | Not started |

**Architecture note:** the production query-time selector (Rust `candidate-selection` /
`indexer-selection`, weighted-product model, slashable-GRT as a *positive* input) already
encodes most of the above. The strongest single move is to converge on that audited
library rather than maintain a divergent Python re-implementation with an inverted econ
signal.
