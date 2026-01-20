"""
Loads pre-computed indexer scores from BigQuery.

Scores are computed daily by a CronJob (jobs/compute_scores/) and written to the
indexer_scores table. IISA reads these scores on startup using DataManager.load_scores().
"""

from .manager import DataManager

__all__ = ["DataManager"]
