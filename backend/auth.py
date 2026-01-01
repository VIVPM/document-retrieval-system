"""
auth.py — password hashing, JWT issue/verify, and the current-user dependency.

The security boundary for chat data is Postgres, not Pinecone: every chat
endpoint resolves the caller to a user_id here, then checks
`chat.user_id == user_id` before it touches a namespace.
"""

import hashlib
import os
import secrets
from datetime import datetime, timedelta, timezone

import bcrypt
import jwt
from fastapi import Header, HTTPException

_secret = os.getenv("JWT_SECRET")
if not _secret:
    raise ValueError(
        "JWT_SECRET is not set. Add a long random string to backend/.env:\n"
        "  JWT_SECRET=$(python -c \"import secrets; print(secrets.token_urlsafe(48))\")"
    )
JWT_SECRET: str = _secret

JWT_ALGORITHM = "HS256"
# Access token is short-lived because it is stateless and cannot be revoked —
# a leaked one expires fast. Staying logged in is the refresh token's job.
ACCESS_TTL_MINUTES = int(os.getenv("ACCESS_TTL_MINUTES", "60"))
# Refresh token is long-lived but revocable: stored server-side as a hash, it
# silently mints new access tokens and is what /logout deletes.
REFRESH_TTL_DAYS = int(os.getenv("REFRESH_TTL_DAYS", "14"))

MAX_LOGIN_FAILURES = 5
LOGIN_LOCKOUT_MINUTES = 15


# ── Passwords ─────────────────────────────────────────────────────────────────

def hash_password(password: str) -> str:
    return bcrypt.hashpw(password.encode("utf-8"), bcrypt.gensalt()).decode("utf-8")


def verify_password(password: str, hashed: str) -> bool:
    """Verify against bcrypt, falling back to legacy SHA-256 so old rows still log in."""
    try:
        return bcrypt.checkpw(password.encode("utf-8"), hashed.encode("utf-8"))
    except (ValueError, TypeError):
        # Not a bcrypt hash — a pre-bcrypt row. Caller re-hashes on success.
        return hashlib.sha256(password.encode("utf-8")).hexdigest() == hashed


def needs_rehash(hashed: str) -> bool:
    return not hashed.startswith("$2b$")


# ── Tokens ────────────────────────────────────────────────────────────────────

def create_token(user_id: int, username: str) -> str:
    payload = {
        "user_id": user_id,
        "username": username,
        "exp": datetime.now(timezone.utc) + timedelta(minutes=ACCESS_TTL_MINUTES),
    }
    return jwt.encode(payload, JWT_SECRET, algorithm=JWT_ALGORITHM)


def make_refresh_token() -> tuple[str, str]:
    """Return (raw_token, token_hash). The raw goes to the client once; only
    the hash is stored, so a DB leak can't be replayed as a valid token."""
    raw = secrets.token_urlsafe(32)
    return raw, hash_refresh_token(raw)


def hash_refresh_token(raw: str) -> str:
    return hashlib.sha256(raw.encode("utf-8")).hexdigest()


def get_current_user(authorization: str = Header(...)) -> dict:
    """FastAPI dependency. Expects `Authorization: Bearer <token>`."""
    if not authorization.startswith("Bearer "):
        raise HTTPException(status_code=401, detail="Invalid authorization header.")

    token = authorization[len("Bearer "):]
    try:
        payload = jwt.decode(token, JWT_SECRET, algorithms=[JWT_ALGORITHM])
    except jwt.ExpiredSignatureError:
        raise HTTPException(status_code=401, detail="Session expired. Please log in again.")
    except jwt.InvalidTokenError:
        raise HTTPException(status_code=401, detail="Invalid token.")

    return {"user_id": payload["user_id"], "username": payload["username"]}
