"""
Langfuse OpenTelemetry integration for LiveKit Agents.

Provides real-time tracing and observability by configuring an OpenTelemetry
TracerProvider connected to Langfuse Cloud / self-hosted Langfuse, and
registering it with LiveKit's native agent telemetry system.

Reference: https://langfuse.com/integrations/frameworks/livekit
"""

import logging
import os
from typing import Optional
from dotenv import find_dotenv, load_dotenv
from langfuse import Langfuse
from opentelemetry.sdk.trace import TracerProvider
from opentelemetry.util.types import AttributeValue

from livekit.agents.telemetry import set_tracer_provider

logger = logging.getLogger("livekit.langfuse")

load_dotenv(find_dotenv())


def setup_langfuse(
    metadata: Optional[dict[str, AttributeValue]] = None,
    *,
    base_url: Optional[str] = None,
    public_key: Optional[str] = None,
    secret_key: Optional[str] = None,
) -> Optional[TracerProvider]:
    """
    Configure and register Langfuse tracing for LiveKit Agents via OpenTelemetry.

    Args:
        metadata: Optional dictionary of attributes to attach to all spans
                  (e.g., {"langfuse.session.id": room_name}).
        base_url: Langfuse API URL (defaults to LANGFUSE_BASE_URL or LANGFUSE_HOST).
        public_key: Langfuse public key (defaults to LANGFUSE_PUBLIC_KEY).
        secret_key: Langfuse secret key (defaults to LANGFUSE_SECRET_KEY).

    Returns:
        TracerProvider instance if successfully configured, or None if keys are missing.
    """
    public_key = public_key or os.getenv("LANGFUSE_PUBLIC_KEY")
    secret_key = secret_key or os.getenv("LANGFUSE_SECRET_KEY")
    base_url = (
        base_url
        or os.getenv("LANGFUSE_BASE_URL")
        or os.getenv("LANGFUSE_HOST")
        or "https://cloud.langfuse.com"
    )

    if not public_key or not secret_key:
        logger.warning(
            "Langfuse credentials not found (LANGFUSE_PUBLIC_KEY and/or "
            "LANGFUSE_SECRET_KEY missing). Observability tracing is disabled."
        )
        return None

    try:
        trace_provider = TracerProvider()

        # Register tracer provider with LiveKit agent telemetry
        set_tracer_provider(trace_provider, metadata=metadata)

        # Initialize Langfuse with the OTel TracerProvider
        Langfuse(
            public_key=public_key,
            secret_key=secret_key,
            base_url=base_url,
            tracer_provider=trace_provider,
            should_export_span=lambda span: True,
        )

        session_id = metadata.get("langfuse.session.id") if metadata else None
        logger.info(
            "Langfuse tracing successfully initialized (base_url=%s, session_id=%s)",
            base_url,
            session_id,
        )
        return trace_provider

    except Exception as e:
        logger.exception("Failed to initialize Langfuse tracing: %s", e)
        return None
