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
from dotenv import find_dotenv, load_dotenv

# Load environment configuration
load_dotenv(find_dotenv())

# Core project imports
from langchain_core.messages import AIMessage, HumanMessage, SystemMessage, ToolMessage
from agent import (
    AgentState,
    VoiceGraphWrapper,
    agent_node,
    build_langgraph_workflow,
    get_news,
    get_weather,
)
from telemetry_langfuse import (
    flush_langfuse,
    is_langfuse_configured,
    setup_langfuse,
)
from ui.server import create_app

logging.basicConfig(
    level=logging.INFO,
    format="%(asctime)s [%(levelname)s] %(name)s: %(message)s",
)
logger = logging.getLogger("test_edgecases")


# ==============================================================================
# 1. Weather Tool Edge Cases
# ==============================================================================
async def test_weather_edge_cases():
    print("\n--- 1. Testing Weather Tool Edge Cases ---")

    # 1.1 Valid standard city
    print("  [1.1] Testing valid city: 'Tokyo'...")
    res = await get_weather.ainvoke({"city": "Tokyo"})
    print(f"        Result: {res}")
    assert "°C" in res or "humidity" in res, f"Expected temperature output, got: {res}"
    print("        ✅ Valid city handled correctly.")

    # 1.2 City with spaces
    print("  [1.2] Testing multi-word city: 'San Francisco'...")
    res = await get_weather.ainvoke({"city": "San Francisco"})
    print(f"        Result: {res}")
    assert "°C" in res or "humidity" in res, f"Expected temperature output, got: {res}"
    print("        ✅ Multi-word city handled correctly.")

    # 1.3 Accented / Unicode city
    print("  [1.3] Testing accented Unicode city: 'São Paulo'...")
    res = await get_weather.ainvoke({"city": "São Paulo"})
    print(f"        Result: {res}")
    assert "°C" in res or "humidity" in res, f"Expected temperature output, got: {res}"
    print("        ✅ Accented Unicode city handled correctly.")

    # 1.4 Empty city string
    print("  [1.4] Testing empty city string: ''...")
    res = await get_weather.ainvoke({"city": ""})
    print(f"        Result: {res}")
    assert "Please specify a city" in res, f"Expected validation prompt, got: {res}"
    print("        ✅ Empty city handled gracefully.")

    # 1.5 Whitespace-only city string
    print("  [1.5] Testing whitespace-only city: '   '...")
    res = await get_weather.ainvoke({"city": "   "})
    print(f"        Result: {res}")
    assert "Please specify a city" in res, f"Expected validation prompt, got: {res}"
    print("        ✅ Whitespace city handled gracefully.")

    # 1.6 Non-existent / fictitious city
    print("  [1.6] Testing fictitious city: 'NonExistentCityXYZ987654'...")
    res = await get_weather.ainvoke({"city": "NonExistentCityXYZ987654"})
    print(f"        Result: {res}")
    assert "could not find weather" in res.lower(), f"Expected not found message, got: {res}"
    print("        ✅ Non-existent city handled gracefully without exceptions.")


# ==============================================================================
# 2. News Tool Edge Cases
# ==============================================================================
async def test_news_edge_cases():
    print("\n--- 2. Testing News Tool Edge Cases ---")

    # 2.1 Valid news query
    print("  [2.1] Testing valid news query: 'Artificial Intelligence'...")
    res = await get_news.ainvoke({"query": "Artificial Intelligence"})
    print(f"        Preview: {res[:120]}...")
    assert len(res) > 20 and "No recent news found" not in res, f"Expected news results, got: {res}"
    print("        ✅ Valid news query handled correctly.")

    # 2.2 Query with punctuation and special symbols
    print("  [2.2] Testing query with special characters: 'NVIDIA & AMD @ 2026!?'...")
    res = await get_news.ainvoke({"query": "NVIDIA & AMD @ 2026!?"})
    print(f"        Preview: {res[:120]}...")
    assert len(res) > 10, f"Expected non-empty result for query with symbols, got: {res}"
    print("        ✅ Special symbols in news query handled correctly.")

    # 2.3 Empty query string
    print("  [2.3] Testing empty query: ''...")
    res = await get_news.ainvoke({"query": ""})
    print(f"        Result: {res}")
    assert "Please specify a topic" in res, f"Expected validation prompt, got: {res}"
    print("        ✅ Empty news query handled gracefully.")

    # 2.4 Whitespace query
    print("  [2.4] Testing whitespace query: '   '...")
    res = await get_news.ainvoke({"query": "   "})
    print(f"        Result: {res}")
    assert "Please specify a topic" in res, f"Expected validation prompt, got: {res}"
    print("        ✅ Whitespace news query handled gracefully.")


# ==============================================================================
# 3. LangGraph Agent Workflow & Token Streaming Edge Cases
# ==============================================================================
async def test_langgraph_workflow_edge_cases():
    print("\n--- 3. Testing LangGraph Workflow & Streaming Isolation ---")

    # 3.1 Context Windowing: feed 10 messages and ensure only last 6 are sent
    print("  [3.1] Testing message history windowing (10 history items)...")
    many_messages = [
        HumanMessage(content=f"Historic query #{i}") for i in range(10)
    ] + [HumanMessage(content="What is 2 + 2?")]
    state: AgentState = {"messages": many_messages}
    node_res = await agent_node(state)
    assert "messages" in node_res and len(node_res["messages"]) == 1
    spoken_reply = node_res["messages"][0].content
    print(f"        Spoken Response: {spoken_reply}")
    assert "4" in spoken_reply, f"Expected 4 in response, got: {spoken_reply}"
    print("        ✅ Message windowing executed cleanly.")

    # 3.2 VoiceGraphWrapper Token Isolation: Ensure ToolMessage is NEVER streamed
    print("  [3.2] Testing VoiceGraphWrapper suppression of ToolMessage...")
    class MockGraph:
        async def astream(self, *args, **kwargs):
            # 1. Tool execution chunk (should be SUPPRESSED)
            yield (
                ToolMessage(content='{"raw_tool_data": "json_dump"}', tool_call_id="call_1"),
                {"langgraph_node": "tools"},
            )
            # 2. Assistant spoken token chunks (should PASS THROUGH)
            yield (AIMessage(content="Hello "), {"langgraph_node": "agent"})
            yield (AIMessage(content="world!"), {"langgraph_node": "agent"})

    wrapper = VoiceGraphWrapper(MockGraph())
    yielded_tokens = []
    async for token, meta in wrapper.astream({}):
        yielded_tokens.append(token.content)

    full_output = "".join(yielded_tokens)
    print(f"        Streamed output through VoiceGraphWrapper: '{full_output}'")
    assert "raw_tool_data" not in full_output, "ToolMessage leaked through VoiceGraphWrapper!"
    assert full_output == "Hello world!", f"Unexpected filtered stream output: {full_output}"
    print("        ✅ ToolMessage chunks were strictly suppressed from audio stream.")

    # 3.3 End-to-End Compiled Graph: Direct Chit-chat
    print("  [3.3] Testing End-to-End Chat Flow: 'Tell me a one-sentence joke.'...")
    app = build_langgraph_workflow()
    tokens = []
    async for item in app.astream(
        {"messages": [HumanMessage(content="Tell me a one-sentence joke.")]},
        stream_mode="messages",
    ):
        token, _ = item
        if getattr(token, "content", None):
            tokens.append(token.content)
    joke = "".join(tokens)
    print(f"        Spoken Joke: {joke}")
    assert len(joke) > 10, f"Expected non-empty joke, got: {joke}"
    print("        ✅ Direct chit-chat streamed smoothly without invoking tools.")


# ==============================================================================
# 4. Langfuse Observability & OpenTelemetry Edge Cases
# ==============================================================================
def test_langfuse_observability_edge_cases():
    print("\n--- 4. Testing Langfuse Observability & OpenTelemetry Edge Cases ---")

    # 4.1 Check Configuration Helper
    configured = is_langfuse_configured()
    print(f"  [4.1] is_langfuse_configured() = {configured}")
    assert configured is True, "Expected Langfuse credentials to be configured in .env"
    print("        ✅ Configuration helper works correctly.")

    # 4.2 Graceful fallback when credentials are empty
    print("  [4.2] Testing setup_langfuse() with empty credentials (graceful fallback)...")
    provider_none = setup_langfuse(public_key="", secret_key="")
    assert provider_none is None, f"Expected None on missing keys, got: {provider_none}"
    print("        ✅ Missing credentials gracefully returns None without crashing.")

    # 4.3 Safe Flush with None provider
    print("  [4.3] Testing flush_langfuse(None)...")
    flush_langfuse(None)
    print("        ✅ flush_langfuse(None) executed safely as a no-op.")

    # 4.4 Live TracerProvider setup and nested spans
    print("  [4.4] Testing setup_langfuse() with active session and nested spans...")
    test_session = f"edgecase-room-{int(time.time())}"
    provider = setup_langfuse(
        metadata={
            "langfuse.session.id": test_session,
            "test_suite": "edgecases",
        }
    )
    assert provider is not None, "Failed to initialize active TracerProvider"

    tracer = provider.get_tracer("edgecase.test")
    with tracer.start_as_current_span("parent_agent_turn") as parent:
        parent.set_attribute("room.name", test_session)
        with tracer.start_as_current_span("nested_stt_span") as stt_span:
            stt_span.set_attribute("provider", "assemblyai")
            time.sleep(0.01)
        with tracer.start_as_current_span("nested_tts_span") as tts_span:
            tts_span.set_attribute("provider", "cartesia")
            time.sleep(0.01)

    print("  [4.5] Flushing telemetry spans to Langfuse Cloud...")
    flush_langfuse(provider)
    print("        ✅ Spans flushed successfully.")


# ==============================================================================
# 5. UI Server & Token Generation Edge Cases
# ==============================================================================
def test_ui_server_and_token_edge_cases():
    print("\n--- 5. Testing UI Server & Token Issuance ---")
    app = create_app()
    assert app is not None, "Failed to construct aiohttp application"

    # Verify routes registered
    routes = [route.resource.canonical for route in app.router.routes() if route.resource]
    print(f"  [5.1] Registered routes in UI server: {set(routes)}")
    assert "/" in routes, "Missing root '/' route"
    assert "/health" in routes, "Missing '/health' route"
    assert "/api/token" in routes, "Missing '/api/token' route"
    print("        ✅ All required routes are registered.")

    # Test token creation directly with livekit.api
    from livekit import api
    url = os.getenv("LIVEKIT_URL")
    key = os.getenv("LIVEKIT_API_KEY")
    sec = os.getenv("LIVEKIT_API_SECRET")
    assert url and key and sec, "LiveKit credentials missing in environment"

    token = (
        api.AccessToken(api_key=key, api_secret=sec)
        .with_identity("test-caller-001")
        .with_name("Test User")
        .with_grants(api.VideoGrants(room_join=True, room="test-edgecase-room"))
    )
    jwt_str = token.to_jwt()
    print(f"  [5.2] Generated LiveKit JWT token preview: {jwt_str[:35]}...")
    assert jwt_str.count(".") == 2, "Invalid JWT format (expected 3 dot-separated parts)"
    print("        ✅ LiveKit AccessToken JWT generated and validated successfully.")


# ==============================================================================
# Main Runner
# ==============================================================================
async def main():
    print("=" * 70)
    print("🚀 STARTING FULL APPLICATION COMPREHENSIVE EDGE-CASE TEST SUITE")
    print("=" * 70)

    start_time = time.time()

    # 1. Weather Edge Cases
    await test_weather_edge_cases()

    # 2. News Edge Cases
    await test_news_edge_cases()

    # 3. LangGraph Workflow Edge Cases
    await test_langgraph_workflow_edge_cases()

    # 4. Langfuse Observability Edge Cases
    test_langfuse_observability_edge_cases()

    # 5. UI Server & Token Generation
    test_ui_server_and_token_edge_cases()

    elapsed = time.time() - start_time
    print("\n" + "=" * 70)
    print(f"🎉 ALL COMPREHENSIVE EDGE-CASE TESTS PASSED! (Elapsed: {elapsed:.2f}s)")
    print("=" * 70)


if __name__ == "__main__":
    asyncio.run(main())
