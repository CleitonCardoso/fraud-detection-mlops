"""API key authentication for sensitive endpoints (/predict, /agent/query).

Public endpoints (/health, /metrics/) intentionally remain unauthenticated so
liveness probes and Prometheus scrapes keep working.
"""

import os

from fastapi import Header, HTTPException, status


def _allowed_keys() -> set[str]:
    raw = os.getenv("FRAUD_API_KEY", "")
    return {k.strip() for k in raw.split(",") if k.strip()}


def verify_api_key(x_api_key: str | None = Header(default=None)) -> None:
    """Reject request when FRAUD_API_KEY is set and the header is missing or invalid.

    When FRAUD_API_KEY is unset (typical for local dev/tests), auth is disabled.
    """
    keys = _allowed_keys()
    if not keys:
        return
    if x_api_key is None or x_api_key not in keys:
        raise HTTPException(
            status_code=status.HTTP_401_UNAUTHORIZED,
            detail="Invalid or missing API key",
        )
