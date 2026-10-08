"""Smoke check: tracing stays disabled without Langfuse credentials."""
import os
from unittest.mock import patch

import observability


def test_no_langfuse_credentials():
    with patch.dict(os.environ, {
        "LANGFUSE_PUBLIC_KEY": "", "LANGFUSE_SECRET_KEY": "",
        "OTHER_OTLP_ENDPOINT": "https://example.invalid/otlp",
    }):
        observability.init_observability()
        assert observability._llm_tracer is None


if __name__ == "__main__":
    test_no_langfuse_credentials()
