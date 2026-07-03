"""
ratelimit.py
Per-tier rate limiting for the Avarent API.

Tier → request limit mapping:
    owner   unlimited (10 000 / minute effectively)
    pro     120 / minute
    free     30 / minute

Root cause of the ContextVar approach failing:
  slowapi calls _dynamic_limit() BEFORE it calls key_func, so
  setting the ContextVar inside key_func is too late.

Fix: subclass Limiter and override _check_request_limit to set the
ContextVar first, before any limit evaluation happens.
"""

import contextvars
from fastapi import Request
from slowapi import Limiter

from avarent.auth import VALID_KEYS

TIER_LIMITS: dict[str, str] = {
    "owner": "10000/minute",
    "pro":   "120/minute",
    "free":  "30/minute",
}
DEFAULT_LIMIT = "30/minute"

# Holds the current request for the duration of limit evaluation
_request_ctx: contextvars.ContextVar[Request | None] = contextvars.ContextVar(
    "_request_ctx", default=None
)


def _key_from_api_key(request: Request) -> str:
    """Rate-limit bucket — one counter per API key."""
    return request.headers.get("x-api-key") or request.client.host


def _dynamic_limit() -> str:
    """
    Called by slowapi (no args) to decide which limit to apply.
    Reads the request that TieredLimiter stored before evaluation started.
    """
    request = _request_ctx.get()
    if request is None:
        return DEFAULT_LIMIT

    api_key = request.headers.get("x-api-key", "")
    client  = VALID_KEYS.get(api_key, {})
    tier    = client.get("tier", "free")
    return TIER_LIMITS.get(tier, DEFAULT_LIMIT)


class TieredLimiter(Limiter):
    """
    Subclasses Limiter to set the ContextVar BEFORE slowapi evaluates limits.
    This ensures _dynamic_limit() can see the request when it runs.
    """

    def _check_request_limit(self, request, endpoint_func, is_async=False):
        _request_ctx.set(request)          # set FIRST, before limit evaluation
        super()._check_request_limit(request, endpoint_func, is_async)


# Shared instance imported by main.py
limiter = TieredLimiter(key_func=_key_from_api_key)
