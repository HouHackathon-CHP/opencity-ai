# SPDX-License-Identifier: Apache-2.0
"""Security and gateway authentication dependencies."""

import hmac
from typing import Annotated

from fastapi import Header, HTTPException

from app.core.config import INTERNAL_SECRET


def internal_caller(
    x_internal_secret: Annotated[str | None, Header()] = None,
    x_dxos_tenant: Annotated[str | None, Header()] = None,
    x_dxos_user: Annotated[str | None, Header()] = None,
) -> str:
    """Validate internal microservice calls via shared HMAC secret if configured."""
    if INTERNAL_SECRET and (not x_internal_secret or not hmac.compare_digest(x_internal_secret, INTERNAL_SECRET)):
        raise HTTPException(status_code=403, detail="Gateway access required")
    return x_dxos_tenant or "default"
