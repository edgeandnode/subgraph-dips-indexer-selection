"""
Tests for `_fetch_single_graph_node_version_async`, the per-indexer
fetcher that POSTs a GraphQL `version` query to <url>/status.

These tests stub aiohttp's session.post via a small inline fake so the
fetcher's success / failure / retry handling can be exercised without
running an HTTP server.
"""

import asyncio
import sys
from pathlib import Path

import aiohttp
import pytest

jobs_path = Path(__file__).parent.parent / "cronjobs" / "compute_scores"
sys.path.insert(0, str(jobs_path))

from processing import _fetch_single_graph_node_version_async  # noqa: E402


class _FakeResponse:
    def __init__(self, body, status=200):
        self._body = body
        self.status = status
        self.request_info = None
        self.history = ()

    async def json(self):
        return self._body

    def raise_for_status(self):
        if self.status >= 400 and self.status < 500:
            raise aiohttp.ClientResponseError(
                self.request_info,
                self.history,
                status=self.status,
                message="client error",
            )

    async def __aenter__(self):
        return self

    async def __aexit__(self, *a):
        return False


class _FakeSession:
    """Stub aiohttp session: returns scripted responses for each .post call."""

    def __init__(self, responses):
        self._responses = list(responses)
        self.calls = []

    def post(self, url, json=None, timeout=None):
        self.calls.append({"url": url, "json": json})
        return self._responses.pop(0)


def _run(coro):
    return asyncio.get_event_loop().run_until_complete(coro)


@pytest.fixture
def semaphore():
    return asyncio.Semaphore(10)


def test_fetch_parses_version_and_commit(semaphore):
    # Arrange — happy path: graph-node returns the standard shape.
    session = _FakeSession(
        [_FakeResponse({"data": {"version": {"version": "0.40.1", "commit": "abc123"}}})]
    )

    # Act
    result = _run(
        _fetch_single_graph_node_version_async(
            session, indexer="0xaaa", url="http://i.example", semaphore=semaphore
        )
    )

    # Assert — record carries indexer ID plus both fields verbatim.
    assert result == {
        "indexer": "0xaaa",
        "graph_node_version": "0.40.1",
        "graph_node_commit": "abc123",
    }
    # And the URL was assembled correctly with /status appended.
    assert session.calls[0]["url"] == "http://i.example/status"
    assert session.calls[0]["json"] == {"query": "{ version { version commit } }"}


def test_fetch_handles_missing_version_field(semaphore):
    # Arrange — indexer answered with an empty `data` envelope, e.g.
    # a graph-node build that doesn't have the version query enabled.
    session = _FakeSession([_FakeResponse({"data": {}})])

    # Act
    result = _run(
        _fetch_single_graph_node_version_async(
            session, indexer="0xbbb", url="http://i.example", semaphore=semaphore
        )
    )

    # Assert — fields fall through to None; the filter decides the fate.
    assert result == {
        "indexer": "0xbbb",
        "graph_node_version": None,
        "graph_node_commit": None,
    }


def test_fetch_handles_null_data_envelope(semaphore):
    # Arrange — a GraphQL error response where `data` is null.
    session = _FakeSession([_FakeResponse({"errors": [{"message": "boom"}]})])

    # Act
    result = _run(
        _fetch_single_graph_node_version_async(
            session, indexer="0xccc", url="http://i.example", semaphore=semaphore
        )
    )

    # Assert — no crash, both fields None.
    assert result["graph_node_version"] is None
    assert result["graph_node_commit"] is None


def test_fetch_trims_trailing_slash_from_url(semaphore):
    # Arrange — indexer URLs in the network subgraph occasionally carry
    # a trailing slash; the joiner must not produce a doubled slash.
    session = _FakeSession(
        [_FakeResponse({"data": {"version": {"version": "0.40.0", "commit": "x"}}})]
    )

    # Act
    _run(
        _fetch_single_graph_node_version_async(
            session,
            indexer="0xddd",
            url="http://i.example/",
            semaphore=semaphore,
        )
    )

    # Assert — single slash before /status.
    assert session.calls[0]["url"] == "http://i.example/status"
