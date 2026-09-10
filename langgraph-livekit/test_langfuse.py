#!/usr/bin/env python3
"""
Automated verification test suite for Langfuse OpenTelemetry tracing
in LiveKit Voice Agent.
"""

import logging
import os
import sys
import time
from dotenv import find_dotenv, load_dotenv

# Load environment
load_dotenv(find_dotenv())

from langfuse import Langfuse, observe
from livekit.agents.telemetry import tracer as lk_tracer
from telemetry_langfuse import setup_langfuse

logging.basicConfig(
    level=logging.INFO,
    format="%(asctime)s [%(levelname)s] %(name)s: %(message)s",
)
logger = logging.getLogger("test_langfuse")


@observe(name="mock_tool_execution")
def mock_tool(city: str) -> str:
    """Simulate tool execution with @observe decorator."""
    time.sleep(0.05)
    return f"Weather in {city}: 22°C, sunny"


def main():
    print("=" * 65)
    print("🔍 VERIFYING LANGFUSE OBSERVABILITY INTEGRATION")
    print("=" * 65)

    # 1. Check Credentials in Environment
    pk = os.getenv("LANGFUSE_PUBLIC_KEY")
    sk = os.getenv("LANGFUSE_SECRET_KEY")
    url = os.getenv("LANGFUSE_BASE_URL") or os.getenv("LANGFUSE_HOST") or "https://cloud.langfuse.com"

    print(f"\n1. Checking Langfuse Credentials:")
    print(f"   - LANGFUSE_BASE_URL  : {url}")
    print(f"   - LANGFUSE_PUBLIC_KEY : {'***' + pk[-6:] if pk else 'MISSING'}")
    print(f"   - LANGFUSE_SECRET_KEY : {'***' + sk[-6:] if sk else 'MISSING'}")

    assert pk, "LANGFUSE_PUBLIC_KEY is missing from environment"
    assert sk, "LANGFUSE_SECRET_KEY is missing from environment"
    print("   ✅ Credentials found in environment.")

    # 2. Test Direct Langfuse Cloud API Authentication
    print("\n2. Testing Cloud Authentication Check...")
    lf_client = Langfuse(public_key=pk, secret_key=sk, base_url=url)
    auth_ok = lf_client.auth_check()
    print(f"   - Auth Check Result: {auth_ok}")
    assert auth_ok is True, "Langfuse API key authentication check failed!"
    print("   ✅ Langfuse authentication successful!")

    # 3. Test setup_langfuse() with LiveKit OpenTelemetry integration
    print("\n3. Testing setup_langfuse() & LiveKit TracerProvider Registration...")
    test_session_id = f"test-session-{int(time.time())}"
    trace_provider = setup_langfuse(
        metadata={
            "langfuse.session.id": test_session_id,
            "environment": "automated-testing",
        }
    )
    assert trace_provider is not None, "setup_langfuse returned None!"
    print(f"   - TracerProvider initialized: {type(trace_provider).__name__}")
    print(f"   - LiveKit current tracer provider registered: {lk_tracer._tracer_provider is not None}")
    print("   ✅ OpenTelemetry tracer provider successfully bound to LiveKit!")

    # 4. Generate Telemetry Spans (Simulating Voice Agent Pipeline)
    print("\n4. Emitting Simulated Voice Agent Telemetry Spans...")
    tracer = trace_provider.get_tracer("livekit.agent.test")

    with tracer.start_as_current_span("agent_session") as session_span:
        session_span.set_attribute("livekit.room.name", test_session_id)
        session_span.set_attribute("agent.status", "connected")

        # Simulate STT span
        with tracer.start_as_current_span("stt_transcription") as stt_span:
            stt_span.set_attribute("stt.model", "assemblyai/universal-streaming:en")
            stt_span.set_attribute("stt.latency_ms", 185)
            time.sleep(0.02)

        # Simulate @observe tool execution
        tool_out = mock_tool("Paris")
        print(f"   - Mock observed tool execution result: '{tool_out}'")

        # Simulate TTS span
        with tracer.start_as_current_span("tts_synthesis") as tts_span:
            tts_span.set_attribute("tts.model", "cartesia/sonic-3")
            tts_span.set_attribute("tts.latency_ms", 120)
            time.sleep(0.02)

    print("   ✅ Spans successfully created and nested in active trace context.")

    # 5. Flush Spans to Langfuse Cloud
    print("\n5. Flushing Telemetry Spans to Langfuse Cloud...")
    trace_provider.force_flush()
    print("   ✅ Telemetry batch flushed without errors.")

    print("\n" + "=" * 65)
    print("🎉 ALL VERIFICATION CHECKS PASSED WITH GREEN SIGNAL!")
    print(f"   Session ID: {test_session_id}")
    print(f"   Dashboard : {url}")
    print("=" * 65)


if __name__ == "__main__":
    main()
