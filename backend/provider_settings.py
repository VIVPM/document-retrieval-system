"""Encrypted account credentials; public settings expose only the last four characters."""

import base64
import hashlib
import logging

from cryptography.fernet import Fernet, InvalidToken

from auth import JWT_SECRET
from db.database import SessionLocal
from db.models import AccountProvider

log = logging.getLogger(__name__)
# Match the ecommerce app: rotating JWT_SECRET requires saving credentials again.
_fernet = Fernet(base64.urlsafe_b64encode(hashlib.sha256(JWT_SECRET.encode()).digest()))


def _decrypt(value):
    if not value:
        return None
    try:
        return _fernet.decrypt(value.encode()).decode()
    except InvalidToken:
        log.warning("Saved provider credentials cannot be decrypted; save them again")
        return None


def load_settings(user_id):
    with SessionLocal() as db:
        row = db.get(AccountProvider, user_id)
        return {
            "selected": row.selected if row else "FREE",
            "gemini_key": _decrypt(row.gemini_key_enc) if row else None,
            "cloudflare_token": _decrypt(row.cloudflare_token_enc) if row else None,
            "cloudflare_account_id": row.cloudflare_account_id if row else None,
        }


def own_credentials(settings):
    selected = settings["selected"]
    if selected == "GEMINI" and settings["gemini_key"]:
        return selected, settings["gemini_key"], ""
    if selected == "CLOUDFLARE" and settings["cloudflare_token"] and settings["cloudflare_account_id"]:
        return selected, settings["cloudflare_token"], settings["cloudflare_account_id"]
    if selected != "FREE":
        raise ValueError("Saved provider credentials are unavailable. Save them again in Settings.")
    return None, "", ""


def public_settings(settings):
    return {
        "selected": settings["selected"],
        "gemini": {"saved": bool(settings["gemini_key"]),
                   "last4": settings["gemini_key"][-4:] if settings["gemini_key"] else None},
        "cloudflare": {"saved": bool(settings["cloudflare_token"] and settings["cloudflare_account_id"]),
                       "last4": settings["cloudflare_token"][-4:] if settings["cloudflare_token"] else None,
                       "account_id": settings["cloudflare_account_id"]},
    }


def save_credentials(user_id, provider, key, account_id=""):
    if provider not in ("GEMINI", "CLOUDFLARE"):
        raise ValueError("Unknown provider")
    with SessionLocal() as db:
        row = db.get(AccountProvider, user_id) or AccountProvider(user_id=user_id)
        encrypted = _fernet.encrypt(key.encode()).decode()
        if provider == "GEMINI":
            row.gemini_key_enc = encrypted
        else:
            row.cloudflare_token_enc = encrypted
            row.cloudflare_account_id = account_id
        row.selected = provider
        db.add(row)
        db.commit()


def select_provider(user_id, provider):
    if provider not in ("FREE", "GEMINI", "CLOUDFLARE"):
        raise ValueError("Unknown provider")
    with SessionLocal() as db:
        row = db.get(AccountProvider, user_id) or AccountProvider(user_id=user_id)
        if provider == "GEMINI" and not _decrypt(row.gemini_key_enc):
            raise ValueError("Save a Gemini key before selecting Gemini")
        if provider == "CLOUDFLARE" and not (row.cloudflare_account_id and _decrypt(row.cloudflare_token_enc)):
            raise ValueError("Save Cloudflare credentials before selecting Cloudflare")
        row.selected = provider
        db.add(row)
        db.commit()


def remove_credentials(user_id, provider):
    if provider not in ("GEMINI", "CLOUDFLARE"):
        raise ValueError("Unknown provider")
    with SessionLocal() as db:
        row = db.get(AccountProvider, user_id)
        if row is None:
            return
        if provider == "GEMINI":
            row.gemini_key_enc = None
        else:
            row.cloudflare_token_enc = None
            row.cloudflare_account_id = None
        if row.selected == provider:
            row.selected = "FREE"
        db.commit()
