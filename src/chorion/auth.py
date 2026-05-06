"""
Chorion authentication — JWT validation for service-to-service auth.

HatChat mints short-lived JWTs with claims:
    { sub: userId, role: userRole, iss: "hatchat", aud: "chorion" }

Chorion validates these tokens using a shared secret.
"""

from __future__ import annotations

import os
import logging
from dataclasses import dataclass
from typing import Optional

import jwt
from fastapi import HTTPException, Request, status

logger = logging.getLogger(__name__)

SERVICE_SECRET = os.environ.get("CHORION_SERVICE_SECRET", "")
ISSUER = "hatchat"
AUDIENCE = "chorion"


@dataclass(frozen=True)
class ActorClaims:
    """Validated claims from a service JWT."""
    user_id: str
    role: str  # operator, researcher, auditor


def _extract_token(request: Request) -> Optional[str]:
    """Extract bearer token from Authorization header."""
    auth = request.headers.get("authorization", "")
    if auth.startswith("Bearer "):
        return auth[7:]
    return None


def validate_service_jwt(request: Request) -> ActorClaims:
    """
    Validate a service JWT from HatChat.

    Raises HTTPException 401 if token is missing or invalid.
    """
    token = _extract_token(request)
    if not token:
        raise HTTPException(
            status_code=status.HTTP_401_UNAUTHORIZED,
            detail="Missing authorization token",
        )

    if not SERVICE_SECRET:
        raise HTTPException(
            status_code=status.HTTP_503_SERVICE_UNAVAILABLE,
            detail="Chorion service secret not configured",
        )

    try:
        payload = jwt.decode(
            token,
            SERVICE_SECRET,
            algorithms=["HS256"],
            issuer=ISSUER,
            audience=AUDIENCE,
        )
    except jwt.ExpiredSignatureError:
        raise HTTPException(
            status_code=status.HTTP_401_UNAUTHORIZED,
            detail="Token expired",
        )
    except jwt.InvalidTokenError as e:
        logger.warning("Invalid JWT: %s", e)
        raise HTTPException(
            status_code=status.HTTP_401_UNAUTHORIZED,
            detail="Invalid token",
        )

    user_id = payload.get("sub")
    role = payload.get("role", "auditor")

    if not user_id:
        raise HTTPException(
            status_code=status.HTTP_401_UNAUTHORIZED,
            detail="Token missing subject",
        )

    return ActorClaims(user_id=user_id, role=role)


def require_role(*allowed_roles: str):
    """
    FastAPI dependency that validates JWT and checks role.

    Usage:
        @router.post("/models", dependencies=[Depends(require_role("operator"))])
    """
    def dependency(request: Request) -> ActorClaims:
        claims = validate_service_jwt(request)
        if claims.role not in allowed_roles:
            raise HTTPException(
                status_code=status.HTTP_403_FORBIDDEN,
                detail=f"Role '{claims.role}' not permitted; requires one of {allowed_roles}",
            )
        return claims
    return dependency
