"""Optional Langfuse tracing via the LangChain callback interface.

Tracing is enabled only when LANGFUSE_PUBLIC_KEY and LANGFUSE_SECRET_KEY are set.
Any import or initialisation failure is logged and tracing is disabled, so a
broken observability backend can never take the API down.
"""

from __future__ import annotations

import logging

from app.config import Settings

logger = logging.getLogger(__name__)


def build_callbacks(settings: Settings) -> list:
    if not settings.langfuse_configured:
        return []
    try:
        from langfuse import Langfuse
        from langfuse.langchain import CallbackHandler

        # Langfuse v3 keeps a process-wide client keyed by public key.
        Langfuse(
            public_key=settings.langfuse_public_key,
            secret_key=settings.langfuse_secret_key.get_secret_value(),
            host=settings.langfuse_host,
        )
        handler = CallbackHandler(public_key=settings.langfuse_public_key)
        logger.info("Langfuse tracing enabled (%s)", settings.langfuse_host)
        return [handler]
    except Exception as exc:  # pragma: no cover - depends on external service/package
        logger.warning("Langfuse tracing disabled: %s", exc)
        return []
