#!/usr/bin/env python3
"""
Diagnostic test script to verify both Weather and News tools independently
and within the end-to-end LangGraph tool-calling agent workflow.
"""

import asyncio
import sys
from pathlib import Path

# Ensure project root is in sys.path
PROJECT_ROOT = Path(__file__).resolve().parent.parent
if str(PROJECT_ROOT) not in sys.path:
    sys.path.insert(0, str(PROJECT_ROOT))

from dotenv import find_dotenv, load_dotenv
from langchain_core.messages import HumanMessage

from src.graph import build_langgraph_workflow
from src.tools import get_news, get_weather

load_dotenv(find_dotenv())


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

    # Test A: Weather Query through LangGraph Agent
    print("\n[Test A] Invoking LangGraph agent with: 'What is the weather in London?'")
    result_a = await app.ainvoke({"messages": [HumanMessage(content="What is the weather in London?")]})
    final_message_a = result_a["messages"][-1].content
    print(f"-> Agent Voice Response:\n{final_message_a}")
    assert len(result_a["messages"]) >= 3, "Expected tool execution in graph history!"
    print("✅ LangGraph Agent successfully called Weather tool and generated response!")

    # Test B: News Query through LangGraph Agent
    print("\n[Test B] Invoking LangGraph agent with: 'Give me latest headlines on NVIDIA'")
    result_b = await app.ainvoke({"messages": [HumanMessage(content="Give me latest headlines on NVIDIA")]})
    final_message_b = result_b["messages"][-1].content
    print(f"-> Agent Voice Response:\n{final_message_b}")
    assert len(result_b["messages"]) >= 3, "Expected tool execution in graph history!"
    print("✅ LangGraph Agent successfully called News tool and generated response!")

    # Test C: General Conversation (No Tools Needed)
    print("\n[Test C] Invoking LangGraph agent with: 'Hello! How are you today?'")
    result_c = await app.ainvoke({"messages": [HumanMessage(content="Hello! How are you today?")]})
    final_message_c = result_c["messages"][-1].content
    print(f"-> Agent Voice Response:\n{final_message_c}")
    assert len(result_c["messages"]) == 2, "General chit-chat should NOT invoke tools!"
    print("✅ LangGraph Agent answered general query directly without invoking tools!")

    print("\n" + "=" * 60)
    print("🎉 ALL TESTS PASSED SUCCESSFULLY!")
    print("=" * 60)


if __name__ == "__main__":
    asyncio.run(main())
