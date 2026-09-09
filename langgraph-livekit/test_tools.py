#!/usr/bin/env python3
"""
Diagnostic test script to verify both Weather and News tools independently
and within the end-to-end LangGraph tool-calling agent workflow.
"""

import asyncio
from dotenv import find_dotenv, load_dotenv

load_dotenv(find_dotenv())

from langchain_core.messages import HumanMessage
from agent import build_langgraph_workflow, get_news, get_weather


async def main():
    print("=" * 60)
    print("🔍 STEP 1: Direct Tool Calling Function Tests")
    print("=" * 60)

    # 1. Test Weather Tool
    print("\n[Testing Weather Tool] Invoking get_weather for 'Tokyo'...")
    weather_result = await get_weather.ainvoke({"city": "Tokyo"})
    print(f"-> Weather Output: {weather_result}")
    assert "°C" in weather_result, "Weather tool failed!"
    print("✅ Weather Tool is WORKING!")

    # 2. Test News Tool
    print("\n[Testing News Tool] Invoking get_news for 'Artificial Intelligence'...")
    news_result = await get_news.ainvoke({"query": "Artificial Intelligence"})
    print(f"-> News Output Preview:\n{news_result[:250]}...")
    assert news_result and "No recent news found" not in news_result, "News tool failed!"
    print("✅ News Tool is WORKING!")

    print("\n" + "=" * 60)
    print("🔄 STEP 2: End-to-End Tool-Calling Agent Workflow Tests")
    print("=" * 60)

    app = build_langgraph_workflow()

    # 3. Weather query
    print("\n[User Query]: 'What is the weather in London right now?'")
    streamed_weather = []
    async for item in app.astream({"messages": [HumanMessage(content="What is the weather in London right now?")]}, stream_mode="messages"):
        token, _ = item
        if getattr(token, "content", None):
            streamed_weather.append(token.content)
    spoken_weather = "".join(streamed_weather)
    print(f"-> Spoken LLM Response: {spoken_weather}")
    assert len(spoken_weather) > 10, "Empty response from agent!"

    # 4. News query
    print("\n[User Query]: 'What are the top news headlines about OpenAI today?'")
    streamed_news = []
    async for item in app.astream({"messages": [HumanMessage(content="What are the top news headlines about OpenAI today?")]}, stream_mode="messages"):
        token, _ = item
        if getattr(token, "content", None):
            streamed_news.append(token.content)
    spoken_news = "".join(streamed_news)
    print(f"-> Spoken LLM Response: {spoken_news}")
    assert len(spoken_news) > 10, "Empty response from agent!"

    # 5. Direct chit-chat query (no tool needed)
    print("\n[User Query]: 'Hello! How are you doing today?'")
    streamed_chat = []
    async for item in app.astream({"messages": [HumanMessage(content="Hello! How are you doing today?")]}, stream_mode="messages"):
        token, _ = item
        if getattr(token, "content", None):
            streamed_chat.append(token.content)
    spoken_chat = "".join(streamed_chat)
    print(f"-> Spoken LLM Response: {spoken_chat}")
    assert len(spoken_chat) > 5, "Empty response from agent!"

    print("\n" + "=" * 60)
    print("🎉 ALL TESTS PASSED! Agent node with tool calls is fully operational.")
    print("=" * 60)


if __name__ == "__main__":
    asyncio.run(main())
