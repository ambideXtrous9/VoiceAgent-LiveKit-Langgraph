"""Langfuse OpenTelemetry setup and trace flushing utilities for LiveKit Voice Agent."""

import logging
import os
from typing import Optional

logger = logging.getLogger("livekit.telemetry")


def is_langfuse_configured() -> bool:
    """Check if required Langfuse environment variables are present."""
    pk = os.getenv("LANGFUSE_PUBLIC_KEY")
    sk = os.getenv("LANGFUSE_SECRET_KEY")
    return bool(pk and sk)


def setup_langfuse(metadata: Optional[dict] = None):
    """
    Initialize Langfuse as an OpenTelemetry span processor and bind it
    to LiveKit Agents telemetry. Returns the TracerProvider instance or None.
    """
    pk = os.getenv("LANGFUSE_PUBLIC_KEY")
    sk = os.getenv("LANGFUSE_SECRET_KEY")
    base_url = os.getenv("LANGFUSE_BASE_URL", "https://us.cloud.langfuse.com")

    if not pk or not sk:
        logger.warning(
            "Langfuse credentials not found (LANGFUSE_PUBLIC_KEY and/or LANGFUSE_SECRET_KEY missing). "
            "Observability tracing is disabled."
        )
        return None

    try:
        from langfuse import Langfuse
        from livekit.agents.telemetry import set_tracer_provider
        from opentelemetry.sdk.trace import TracerProvider

        trace_provider = TracerProvider()
        set_tracer_provider(trace_provider, metadata=metadata)

        langfuse = Langfuse(
            public_key=pk,
            secret_key=sk,
            base_url=base_url,
            tracer_provider=trace_provider,
            should_export_span=lambda span: True,
        )

        session_id = (metadata or {}).get("langfuse.session.id", "unassigned")
        logger.info(
            "Langfuse tracing successfully initialized (base_url=%s, session_id=%s)",
            base_url,
            session_id,
        )
        return trace_provider

    except ImportError as ie:
        logger.warning("Langfuse or OpenTelemetry packages not installed (%s). Tracing disabled.", ie)
        return None
    except Exception as e:
        logger.exception("Failed to initialize Langfuse tracer provider: %s", e)
        return None


def flush_langfuse(trace_provider=None) -> None:
    """
    Safely force-flush buffered OpenTelemetry spans and Langfuse traces.
    Useful for shutdown hooks and session cleanup.
    """
    if trace_provider is None:
        return

    logger.info("Flushing pending telemetry spans to Langfuse...")
    try:
        if hasattr(trace_provider, "force_flush"):
            trace_provider.force_flush()
        try:
            from langfuse import Langfuse
            Langfuse().flush()
        except Exception:
            pass
        logger.info("Langfuse telemetry flush completed successfully.")
    except Exception as e:
        logger.warning("Error during Langfuse trace flush: %s", e)
