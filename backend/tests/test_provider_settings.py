"""Account credentials survive new sessions without exposing plaintext to clients."""

import os
import unittest
from unittest.mock import patch

os.environ.setdefault("JWT_SECRET", "test-secret-not-for-production")
os.environ.setdefault("DATABASE_URL", "postgresql://unused:unused@localhost/unused")

from sqlalchemy import create_engine
from sqlalchemy.orm import sessionmaker

import provider_settings as settings
from db.models import AccountProvider


class ProviderSettingsTest(unittest.TestCase):
    def setUp(self):
        self.engine = create_engine("sqlite://")
        AccountProvider.__table__.create(self.engine)
        self.sessions = sessionmaker(bind=self.engine)
        self.patch = patch.object(settings, "SessionLocal", self.sessions)
        self.patch.start()
        self.addCleanup(self.patch.stop)
        self.addCleanup(self.engine.dispose)

    def test_save_persists_encrypted_and_returns_only_last_four(self):
        key = "test-gemini-secret-1234"
        settings.save_credentials(1, "GEMINI", key)
        with self.sessions() as db:
            row = db.get(AccountProvider, 1)
            self.assertNotIn(key, row.gemini_key_enc)
        loaded = settings.load_settings(1)
        self.assertEqual(settings.own_credentials(loaded), ("GEMINI", key, ""))
        public = settings.public_settings(loaded)
        self.assertEqual(public["gemini"], {"saved": True, "last4": "1234"})
        self.assertNotIn(key, str(public))
        self.assertEqual(settings.load_settings(2)["selected"], "FREE")

    def test_switch_keeps_both_keys_and_remove_returns_to_free(self):
        settings.save_credentials(1, "GEMINI", "gemini-test-1234")
        settings.save_credentials(1, "CLOUDFLARE", "cloudflare-test-5678", "a" * 32)
        loaded = settings.load_settings(1)
        self.assertEqual(settings.own_credentials(loaded), ("CLOUDFLARE", "cloudflare-test-5678", "a" * 32))
        settings.select_provider(1, "GEMINI")
        settings.remove_credentials(1, "CLOUDFLARE")
        self.assertEqual(settings.load_settings(1)["selected"], "GEMINI")
        settings.remove_credentials(1, "GEMINI")
        self.assertEqual(settings.own_credentials(settings.load_settings(1)), (None, "", ""))

    def test_unavailable_key_never_silently_uses_app_credits(self):
        with self.assertRaises(ValueError):
            settings.select_provider(1, "GEMINI")
        settings.save_credentials(1, "GEMINI", "gemini-test-1234")
        with self.sessions() as db:
            db.get(AccountProvider, 1).gemini_key_enc = "corrupted"
            db.commit()
        with self.assertRaises(ValueError):
            settings.own_credentials(settings.load_settings(1))

    def test_remove_is_account_scoped(self):
        settings.save_credentials(1, "GEMINI", "first-user-secret")
        settings.save_credentials(2, "GEMINI", "second-user-secret")
        settings.remove_credentials(1, "GEMINI")
        self.assertEqual(settings.own_credentials(settings.load_settings(2)),
                         ("GEMINI", "second-user-secret", ""))


if __name__ == "__main__":
    unittest.main()
