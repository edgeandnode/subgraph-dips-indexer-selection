"""HTTP client for pushing scores and sync-status to the IISA service.

The score-computation CronJob and sync-status fetcher use this module to
POST their outputs directly to iisa instead of sharing a filesystem. Each
call is bearer-token authenticated and retried with exponential backoff
until iisa accepts the write or the retry budget is exhausted.

All calls raise IISAPushError on auth failure, validation failure, or
after all retries are exhausted. The caller is expected to fail loud on
exhaustion — the push IS the authoritative write, so a silent drop leaves
iisa serving stale data for up to a full cronjob interval.
"""

from __future__ import annotations

import logging
import os
import time
from typing import Any, Optional

import requests

logger = logging.getLogger(__name__)

# Retry policy: 10 attempts, starting at 0.15s and doubling each time.
# Total wait ~153s before giving up — long enough to ride through an iisa
# rollout (readinessProbe initialDelaySeconds=90, startupProbe up to ~300s)
# but bounded so a stuck iisa fails the cronjob within a few minutes.
RETRY_ATTEMPTS = 10
RETRY_INITIAL_DELAY_SECONDS = 0.15
RETRY_BACKOFF_MULTIPLIER = 2.0

# HTTP timeout per attempt. 10 MiB bodies shouldn't take more than a few
# seconds to ship over the cluster network; pick a generous ceiling.
DEFAULT_TIMEOUT_SECONDS = 30


class IISAPushError(RuntimeError):
    """Raised when a push to iisa fails in a non-retryable way or all retries are exhausted."""


def _auth_header(token: Optional[str]) -> dict[str, str]:
    if not token:
        return {}
    return {"Authorization": f"Bearer {token}"}


def _should_retry(exc: Exception, status_code: Optional[int]) -> bool:
    """Retry transport errors and 5xx responses. Don't retry 4xx — those are our fault."""
    if isinstance(exc, (requests.ConnectionError, requests.Timeout)):
        return True
    if status_code is not None and 500 <= status_code < 600:
        return True
    return False


def _request_with_retry(
    method: str,
    url: str,
    *,
    token: Optional[str],
    json_body: Optional[Any] = None,
    timeout: float = DEFAULT_TIMEOUT_SECONDS,
) -> requests.Response:
    """Send an HTTP request with bearer auth and the shared retry policy.

    Returns the final Response on success. Raises IISAPushError on exhausted retries
    or on any non-retryable response (4xx).
    """
    headers = _auth_header(token)
    if json_body is not None:
        headers["Content-Type"] = "application/json"

    delay = RETRY_INITIAL_DELAY_SECONDS
    last_exc: Optional[Exception] = None
    last_status: Optional[int] = None

    for attempt in range(1, RETRY_ATTEMPTS + 1):
        try:
            response = requests.request(
                method,
                url,
                headers=headers,
                json=json_body,
                timeout=timeout,
            )
            status = response.status_code

            if 200 <= status < 300:
                return response

            # 4xx (except 429) is a client bug — auth, schema, etc. Fail loud, do not retry.
            if 400 <= status < 500 and status != 429:
                body_preview = response.text[:500]
                raise IISAPushError(f"{method} {url} failed with {status}: {body_preview}")

            last_status = status
            last_exc = requests.HTTPError(f"{method} {url} returned {status}", response=response)
            logger.warning(
                "iisa push attempt %d/%d to %s failed with HTTP %d: %s",
                attempt,
                RETRY_ATTEMPTS,
                url,
                status,
                response.text[:200],
            )
        except IISAPushError:
            raise
        except (requests.ConnectionError, requests.Timeout) as exc:
            last_exc = exc
            last_status = None
            logger.warning(
                "iisa push attempt %d/%d to %s failed: %s",
                attempt,
                RETRY_ATTEMPTS,
                url,
                exc,
            )
        except requests.RequestException as exc:
            last_exc = exc
            last_status = None
            logger.warning(
                "iisa push attempt %d/%d to %s failed: %s",
                attempt,
                RETRY_ATTEMPTS,
                url,
                exc,
            )
            if not _should_retry(exc, None):
                break

        if attempt < RETRY_ATTEMPTS:
            time.sleep(delay)
            delay *= RETRY_BACKOFF_MULTIPLIER

    raise IISAPushError(
        f"{method} {url} failed after {RETRY_ATTEMPTS} attempts "
        f"(last_status={last_status}, last_error={last_exc})"
    ) from last_exc


def _require_url(iisa_api_url: str) -> str:
    if not iisa_api_url:
        raise IISAPushError("IISA_API_URL is not set; cannot push to iisa")
    return iisa_api_url.rstrip("/")


def post_scores(iisa_api_url: str, token: Optional[str], payload: list[dict]) -> Any:
    """Push a computed scores array to iisa. Returns the parsed JSON response.

    Raises IISAPushError on failure after all retries exhausted.
    """
    url = f"{_require_url(iisa_api_url)}/scores"
    logger.info("Pushing %d score rows to %s", len(payload), url)
    response = _request_with_retry("POST", url, token=token, json_body=payload)
    return response.json()


def post_sync_status(iisa_api_url: str, token: Optional[str], payload: dict) -> Any:
    """Push a sync-status snapshot to iisa. Returns the parsed JSON response.

    Raises IISAPushError on failure after all retries exhausted.
    """
    url = f"{_require_url(iisa_api_url)}/sync-status"
    logger.info("Pushing sync status for %d indexers to %s", len(payload), url)
    response = _request_with_retry("POST", url, token=token, json_body=payload)
    return response.json()


def get_scores_status(iisa_api_url: str, token: Optional[str]) -> Any:
    """Fetch the current scores status from iisa (used for same-day idempotency).

    Returns {"computed_at": "<iso>"|None}. Raises IISAPushError on failure.
    """
    url = f"{_require_url(iisa_api_url)}/scores/status"
    response = _request_with_retry("GET", url, token=token)
    return response.json()


def get_push_token() -> Optional[str]:
    """Read the bearer token from IISA_PUSH_TOKEN, or return None if unset."""
    token = os.environ.get("IISA_PUSH_TOKEN", "").strip()
    return token or None
