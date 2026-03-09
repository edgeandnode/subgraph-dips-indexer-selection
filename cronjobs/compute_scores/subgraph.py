"""Shared GraphQL pagination helper for querying Graph Network subgraphs.

Both the indexer discovery path (processing.py) and the stake data path
(redpanda.py) need cursor-based pagination over subgraph entities. This
module provides a single implementation so the pagination logic, error
handling, and page-size defaults live in one place.
"""

import logging
from typing import List

import requests

logger = logging.getLogger(__name__)


def paginate_subgraph_query(
    url: str,
    query: str,
    entity: str = "indexers",
    page_size: int = 1000,
) -> List[dict]:
    """Paginate a Graph Network subgraph query using id_gt cursor advancement.

    The query string must use ``$first`` (Int!) and ``$lastId`` (String!)
    variables, and order results by ``id``.  The function fetches pages until
    either an empty page is returned or a page smaller than ``page_size``
    indicates there are no more results.

    Raises ``RuntimeError`` on GraphQL-level errors (``"errors"`` key in
    response).  HTTP and connection errors from ``requests`` propagate
    unhandled so the caller can decide on retry/fallback policy.

    Args:
        url: Subgraph endpoint URL.
        query: GraphQL query string with $first and $lastId variables.
        entity: Top-level field name inside ``data`` to extract results from.
        page_size: Number of entities per page (default 1000, the subgraph
            maximum).

    Returns:
        Flat list of entity dicts across all pages.
    """
    last_id = ""
    all_entities: List[dict] = []

    while True:
        response = requests.post(
            url,
            json={"query": query, "variables": {"first": page_size, "lastId": last_id}},
            timeout=30,
        )
        response.raise_for_status()
        data = response.json()

        if "errors" in data:
            raise RuntimeError(f"GraphQL errors: {data['errors']}")

        page = data.get("data", {}).get(entity, [])
        if not page:
            break

        all_entities.extend(page)
        if len(page) < page_size:
            break
        last_id = page[-1]["id"]

    return all_entities
