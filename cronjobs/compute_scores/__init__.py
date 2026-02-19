"""
CronJob for computing indexer scores.

This job runs daily and computes:
- Latency linear regression coefficients and scores
- Uptime scores
- Success rates
- Stake-to-fees ratios
- Pre-normalized versions of all static metrics

Results are written to a JSON file on the shared PVC.
"""
