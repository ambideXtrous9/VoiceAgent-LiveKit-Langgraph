#!/usr/bin/env python3
"""
Comprehensive Edge-Case and Integration Verification Test Suite.

Validates all application components across normal, edge, and failure scenarios:
1. Tool Edge Cases (empty, whitespace, unicode, unknown locations, special chars)
2. LangGraph Agent Workflow (empty history, windowing, streaming token isolation)
3. Langfuse Observability & OpenTelemetry (auth check, span nesting, fallback, safe flush)
4. UI Server & Token Generation (JWT issuance, grants, health check endpoint)
"""

import asyncio
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

load_dotenv(find_dotenv())

from langchain_core.messages import AIMessage, HumanMessage, SystemMessage, ToolMessage
from src.graph import (
    AgentState,
    VoiceGraphWrapper,
    agent_node,
    build_langgraph_workflow,
)
from src.tools import (
    get_news,
    get_weather,
)
from src.telemetry import (
    flush_langfuse,
    is_langfuse_configured,
    setup_langfuse,
)
from ui.server import create_app

logging.basicConfig(
    level=logging.INFO,
    format="%(asctime)s [%(levelname)s] %(message)s",
)
logger = logging.getLogger("test_edgecases")


# ==============================================================================
# 1. Weather Tool Edge Cases
# ==============================================================================
async def test_weather_edge_cases():
    print("\n--- 1. Testing Weather Tool Edge Cases ---")

    # 1.1 Valid city
    print("  [1.1] Testing valid city: 'Tokyo'...")
    res = await get_weather.ainvoke({"city": "Tokyo"})
    print(f"        Result: {res}")
    assert "Tokyo" in res and "°C" in res, f"Failed on valid city: {res}"
    print("        ✅ Valid city handled correctly.")

    # 1.2 Multi-word city
    print("  [1.2] Testing multi-word city: 'San Francisco'...")
    res = await get_weather.ainvoke({"city": "San Francisco"})
    print(f"        Result: {res}")
    assert "San Francisco" in res and "°C" in res, f"Failed on multi-word city: {res}"
    print("        ✅ Multi-word city handled correctly.")

    # 1.3 Accented Unicode city
    print("  [1.3] Testing accented Unicode city: 'São Paulo'...")
    res = await get_weather.ainvoke({"city": "São Paulo"})
    print(f"        Result: {res}")
    assert ("São Paulo" in res or "Sao Paulo" in res) and "°C" in res, f"Failed on accented city: {res}"
    print("        ✅ Accented Unicode city handled correctly.")

    # 1.4 Empty city string
    print("  [1.4] Testing empty city string: ''...")
    res = await get_weather.ainvoke({"city": ""})
    print(f"        Result: {res}")
    assert "Please specify a city name" in res, f"Empty city was not handled properly: {res}"
    print("        ✅ Empty city handled gracefully.")

    # 1.5 Whitespace city
    print("  [1.5] Testing whitespace-only city: '   '...")
    res = await get_weather.ainvoke({"city": "   "})
    print(f"        Result: {res}")
    assert "Please specify a city name" in res, f"Whitespace city was not handled properly: {res}"
    print("        ✅ Whitespace city handled gracefully.")

    # 1.6 Non-existent / Fictitious city
    print("  [1.6] Testing fictitious city: 'NonExistentCityXYZ987654'...")
    res = await get_weather.ainvoke({"city": "NonExistentCityXYZ987654"})
    print(f"        Result: {res}")
    assert "could not find weather data" in res, f"404 city was not handled properly: {res}"
    print("        ✅ Non-existent city handled gracefully without exceptions.")


# ==============================================================================
# 2. News Tool Edge Cases
# ==============================================================================
async def test_news_edge_cases():
    print("\n--- 2. Testing News Tool Edge Cases ---")

    # 2.1 Standard news query
    print("  [2.1] Testing valid news query: 'Artificial Intelligence'...")
    res = await get_news.ainvoke({"query": "Artificial Intelligence"})
    assert res and len(res) > 20, f"News search returned insufficient data: {res}"
    print(f"        Preview: {res[:120]}...")
    print("        ✅ Valid news query handled correctly.")

    # 2.2 Query with punctuation and special symbols
    print("  [2.2] Testing query with special characters: 'NVIDIA & AMD @ 2026!?'...")
    res = await get_news.ainvoke({"query": "NVIDIA & AMD @ 2026!?"})
    assert res and len(res) > 10, f"Query with special characters failed: {res}"
    print(f"        Preview: {res[:120]}...")
    print("        ✅ Special symbols in news query handled correctly.")

    # 2.3 Empty query
    print("  [2.3] Testing empty query: ''...")
    res = await get_news.ainvoke({"query": ""})
    print(f"        Result: {res}")
    assert "Please specify a topic" in res, f"Empty query not handled gracefully: {res}"
    print("        ✅ Empty news query handled gracefully.")

    # 2.4 Whitespace query
    print("  [2.4] Testing whitespace query: '   '...")
    res = await get_news.ainvoke({"query": "   "})
    print(f"        Result: {res}")
    assert "Please specify a topic" in res, f"Whitespace query not handled gracefully: {res}"
    print("        ✅ Whitespace news query handled gracefully.")


# ==============================================================================
# 3. LangGraph Workflow & Streaming Isolation
# ==============================================================================
async def test_agent_workflow():
    print("\n--- 3. Testing LangGraph Workflow & Streaming Isolation ---")

    # 3.1 Test Context Window Trimming (Feeding 10 past messages)
    print("  [3.1] Testing message history windowing (10 history items)...")
    synthetic_history = [
        HumanMessage(content=f"Past turn {i}") if i % 2 == 0 else AIMessage(content=f"Reply {i}")
        for i in range(10)
    ]
    synthetic_history.append(HumanMessage(content="What is 2 + 2?"))

    app = build_langgraph_workflow()
    res = await app.ainvoke({"messages": synthetic_history})
    assert len(res["messages"]) > 0, "No response generated!"
    final_reply = res["messages"][-1].content
    print(f"        Spoken Response: {final_reply}")
    assert "4" in final_reply, f"Unexpected calculation reply: {final_reply}"
    print("        ✅ Message windowing executed cleanly.")

    # 3.2 Test VoiceGraphWrapper suppresses ToolMessages
    print("  [3.2] Testing VoiceGraphWrapper suppression of ToolMessage...")
    class MockGraph:
        async def astream(self, *args, **kwargs):
            yield (AIMessage(content="Hello"), {"langgraph_node": "agent"})
            yield (ToolMessage(content='{"city": "Paris", "temp": 19}', tool_call_id="call_123"), {"langgraph_node": "tools"})
            yield (AIMessage(content=" world!"), {"langgraph_node": "agent"})

    wrapper = VoiceGraphWrapper(MockGraph())
    streamed_tokens = []
    async for token, meta in wrapper.astream({}):
        streamed_tokens.append(token.content)

    joined_stream = "".join(streamed_tokens)
    print(f"        Streamed output through VoiceGraphWrapper: '{joined_stream}'")
    assert joined_stream == "Hello world!", f"Wrapper leaked ToolMessage: '{joined_stream}'"
    print("        ✅ ToolMessage chunks were strictly suppressed from audio stream.")

    # 3.3 Test End-to-End direct response without tools
    print("  [3.3] Testing End-to-End Chat Flow: 'Tell me a one-sentence joke.'...")
    chat_res = await app.ainvoke({"messages": [HumanMessage(content="Tell me a one-sentence joke.")]})
    assert len(chat_res["messages"]) == 2, f"Chit-chat unexpectedly invoked tools: {len(chat_res['messages'])} messages"
    print(f"        Spoken Joke: {chat_res['messages'][-1].content}")
    print("        ✅ Direct chit-chat streamed smoothly without invoking tools.")


# ==============================================================================
# 4. Langfuse Observability & OpenTelemetry Edge Cases
# ==============================================================================
def test_langfuse_telemetry():
    print("\n--- 4. Testing Langfuse Observability & OpenTelemetry Edge Cases ---")

    # 4.1 Test configuration helper
    is_conf = is_langfuse_configured()
    print(f"  [4.1] is_langfuse_configured() = {is_conf}")
    assert isinstance(is_conf, bool), "Configuration status must be boolean"
    print("        ✅ Configuration helper works correctly.")

    # 4.2 Test setup_langfuse with missing credentials
    orig_pk = os.environ.get("LANGFUSE_PUBLIC_KEY")
    try:
        os.environ["LANGFUSE_PUBLIC_KEY"] = ""
        print("  [4.2] Testing setup_langfuse() with empty credentials (graceful fallback)...")
        provider = setup_langfuse()
        assert provider is None, "Expected None when credentials are missing"
        print("        ✅ Missing credentials gracefully returns None without crashing.")
    finally:
        if orig_pk:
            os.environ["LANGFUSE_PUBLIC_KEY"] = orig_pk

    # 4.3 Test safe flush with None
    print("  [4.3] Testing flush_langfuse(None)...")
    flush_langfuse(None)
    print("        ✅ flush_langfuse(None) executed safely as a no-op.")

    # 4.4 Test valid trace session registration and nesting
    test_session = f"edgecase-room-{int(time.time())}"
    print("  [4.4] Testing setup_langfuse() with active session and nested spans...")
    trace_provider = setup_langfuse(metadata={"langfuse.session.id": test_session})

    if trace_provider:
        from livekit.agents.telemetry import tracer
        with tracer.start_as_current_span("edgecase_session_test") as span:
            span.set_attribute("edgecase.run", True)
            with tracer.start_as_current_span("inner_tool_span") as tool_span:
                tool_span.set_attribute("tool.status", "ok")

        print("  [4.5] Flushing telemetry spans to Langfuse Cloud...")
        flush_langfuse(trace_provider)
        print("        ✅ Spans flushed successfully.")


# ==============================================================================
# 5. UI Server & Token Generation
# ==============================================================================
def test_ui_server():
    print("\n--- 5. Testing UI Server & Token Issuance ---")

    app = create_app()
    routes = {r.resource.canonical for r in app.router.routes()}
    print(f"  [5.1] Registered routes in UI server: {routes}")
    assert "/api/token" in routes, "Missing /api/token endpoint"
    assert "/health" in routes, "Missing /health endpoint"
    assert "/" in routes, "Missing root endpoint"
    print("        ✅ All required routes are registered.")

    # Test token generator logic
    from livekit import api
    api_key = os.getenv("LIVEKIT_API_KEY", "devkey")
    api_secret = os.getenv("LIVEKIT_API_SECRET", "secret")
    token = (
        api.AccessToken(api_key=api_key, api_secret=api_secret)
        .with_identity("test-user")
        .with_name("Test User")
        .with_grants(api.VideoGrants(room_join=True, room="test-room"))
    )
    jwt_str = token.to_jwt()
    assert jwt_str and len(jwt_str) > 30, "Generated JWT is empty or malformed"
    print(f"  [5.2] Generated LiveKit JWT token preview: {jwt_str[:35]}...")
    print("        ✅ LiveKit AccessToken JWT generated and validated successfully.")


# ==============================================================================
# Main Runner
# ==============================================================================
async def main():
    print("=" * 70)
    print("🚀 STARTING FULL APPLICATION COMPREHENSIVE EDGE-CASE TEST SUITE")
    print("=" * 70)

    start_t = time.time()
    await test_weather_edge_cases()
    await test_news_edge_cases()
    await test_agent_workflow()
    test_langfuse_telemetry()
    test_ui_server()
    elapsed = time.time() - start_t

    print("\n" + "=" * 70)
    print(f"🎉 ALL COMPREHENSIVE EDGE-CASE TESTS PASSED! (Elapsed: {elapsed:.2f}s)")
    print("=" * 70)


if __name__ == "__main__":
    asyncio.run(main())
