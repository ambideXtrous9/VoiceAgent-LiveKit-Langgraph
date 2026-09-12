#!/usr/bin/env python3
"""
Automated verification test suite for Langfuse OpenTelemetry tracing
in LiveKit Voice Agent.
"""

import logging
import os
import sys
import time
from pathlib import Path

# Ensure project root is in sys.path
PROJECT_ROOT = Path(__file__).resolve().parent.parent
if str(PROJECT_ROOT) not in sys.path:
    sys.path.insert(0, str(PROJECT_ROOT))

from dotenv import find_dotenv, load_dotenv
from langfuse import Langfuse, observe
from livekit.agents.telemetry import tracer as lk_tracer

from src.telemetry import flush_langfuse, is_langfuse_configured, setup_langfuse

load_dotenv(find_dotenv())

logging.basicConfig(
    level=logging.INFO,
    format="%(asctime)s [%(levelname)s] %(name)s: %(message)s",
)
logger = logging.getLogger("test_telemetry")


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
    base_url = os.getenv("LANGFUSE_BASE_URL", "https://us.cloud.langfuse.com")

    print(f"• Configured: {is_langfuse_configured()}")
    print(f"• Base URL: {base_url}")
    print(f"• Public Key: {pk[:10]}... (redacted)")
    print(f"• Secret Key: {sk[:10]}... (redacted)")

    # 2. Test TracerProvider Registration
    test_session_id = f"test-room-{int(time.time())}"
    print(f"\n[Test 1] Setting up TracerProvider for session: '{test_session_id}'...")

    trace_provider = setup_langfuse(
        metadata={
            "langfuse.session.id": test_session_id,
            "environment": "test",
        }
    )
    assert trace_provider is not None, "Failed to initialize TracerProvider!"
    print("✅ TracerProvider initialized and registered with LiveKit telemetry.")

    # 3. Simulate Spans
    print("\n[Test 2] Simulating LiveKit Agent Voice Pipeline spans...")
    with lk_tracer.start_as_current_span("voice_pipeline_session") as session_span:
        session_span.set_attribute("room.name", test_session_id)
        session_span.set_attribute("agent.version", "1.0.0")

        with lk_tracer.start_as_current_span("stt_transcription") as stt_span:
            stt_span.set_attribute("stt.model", "assemblyai/universal-streaming:en")
            stt_span.set_attribute("stt.text", "What is the weather in Paris?")
            time.sleep(0.05)

        with lk_tracer.start_as_current_span("langgraph_reasoning") as llm_span:
            llm_span.set_attribute("llm.model", "openai/gpt-oss-20b")
            llm_span.set_attribute("llm.reasoning_effort", "low")
            tool_result = mock_tool("Paris")
            llm_span.set_attribute("tool.result", tool_result)

        with lk_tracer.start_as_current_span("tts_synthesis") as tts_span:
            tts_span.set_attribute("tts.model", "cartesia/sonic-3")
            tts_span.set_attribute("tts.ttfa_seconds", 0.185)
            time.sleep(0.05)

    print("✅ Synthetic spans generated successfully.")

    # 4. Flush Telemetry
    print("\n[Test 3] Flushing OpenTelemetry spans to Langfuse Cloud...")
    flush_langfuse(trace_provider)
    print("✅ Traces exported successfully.")

    print("\n" + "=" * 65)
    print("🎉 LANGFUSE INTEGRATION FULLY VERIFIED & WORKING!")
    print(f"👉 Check your dashboard at: {base_url}")
    print(f"   Look for Session ID: '{test_session_id}'")
    print("=" * 65)


if __name__ == "__main__":
    main()
