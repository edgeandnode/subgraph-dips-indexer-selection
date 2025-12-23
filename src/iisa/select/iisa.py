import random
from typing import Optional

from ..typing import IndexerId


def select_one(
    candidate_pool: list[IndexerId],
) -> Optional[IndexerId]:
    """
    Selects a single candidate indexer for indexing a Subgraph deployment.

    :param candidate_pool: A list of candidate indexers to select from.
    :return: The selected indexer ID. If no suitable indexer is found, returns None.
    """
    if len(candidate_pool) == 0:
        return None

    return random.choice(candidate_pool)


def select_many(
    candidate_pool: list[IndexerId],
    n: int,
) -> list[IndexerId]:
    """
    Selects multiple candidate indexers for indexing a Subgraph deployment.

    :param candidate_pool: A list of candidate indexers to select from.
    :param n: The target number of indexers to select.
    :return: A list of selected indexer IDs.
             It can return less than `n` indexers if not enough suitable indexers are found.
    """
    if n <= 0 or len(candidate_pool) == 0:
        return []

    return random.choices(candidate_pool, k=min(n, len(candidate_pool)))
