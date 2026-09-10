#!/usr/bin/env python3
"""
LiveKit Voice Agent with LangGraph Tool-Calling Agent Node.

Architecture:
  User Voice -> LiveKit Room (WebRTC)
    -> LiveKit Agent (STT: AssemblyAI / Deepgram fallback)
      -> LangGraph Tool-Calling Agent Node (Groq: openai/gpt-oss-20b)
        ├── Tool Call: get_weather (OpenWeather 2.5 API)
        └── Tool Call: get_news (DuckDuckGoSearchRun)
      -> Spoken Response Chunks
    -> LiveKit Agent (TTS: Cartesia / Inworld fallback)
  -> User Spoken Audio Output
"""

import asyncio
import logging
import os
import time
from typing import Annotated, TypedDict

import httpx
from dotenv import find_dotenv, load_dotenv
from langchain_community.tools import DuckDuckGoSearchRun
from langchain_community.utilities import DuckDuckGoSearchAPIWrapper
from langchain_core.messages import BaseMessage, HumanMessage, SystemMessage
from langchain_core.tools import tool
from langchain_groq import ChatGroq
from langgraph.graph import START, StateGraph
from langgraph.graph.message import add_messages
from langgraph.prebuilt import ToolNode, tools_condition

from livekit import agents
from livekit.agents import (
    Agent,
    AgentServer,
    AgentSession,
    AgentStateChangedEvent,
    JobContext,
    MetricsCollectedEvent,
    inference,
    metrics,
    room_io,
    stt,
    tts,
)
from livekit.plugins import noise_cancellation, silero
from livekit.plugins.langchain import LLMAdapter

from langfuse import observe
try:
    from telemetry_langfuse import setup_langfuse
except ImportError:
    from langgraph_livekit.telemetry_langfuse import setup_langfuse

# ==============================================================================
# 1. Configuration & Environment Setup
# ==============================================================================

load_dotenv(find_dotenv())

logger = logging.getLogger("livekit.langgraph_agent")

GROQ_MODEL = "openai/gpt-oss-20b"
OPENWEATHER_API_KEY = os.getenv("OPENWEATHER_API_KEY")
GROQ_API_KEY = os.getenv("GROQ_API_KEY")

# ==============================================================================
# 2. Agent Tools (@tool definitions)
# ==============================================================================

ddg_news_tool = DuckDuckGoSearchRun(api_wrapper=DuckDuckGoSearchAPIWrapper(max_results=3, source="news"))
ddg_text_tool = DuckDuckGoSearchRun(api_wrapper=DuckDuckGoSearchAPIWrapper(max_results=3, source="text"))


@tool
@observe(name="get_weather")
async def get_weather(city: str) -> str:
    """Get the current weather and temperature for a given city or location."""
    if not OPENWEATHER_API_KEY:
        logger.warning("OPENWEATHER_API_KEY is not configured.")
        return "Weather service is unavailable because the API key is not configured."

    logger.info("Executing get_weather tool for: '%s'", city)
    try:
        url = "https://api.openweathermap.org/data/2.5/weather"
        async with httpx.AsyncClient(timeout=5.0) as client:
            resp = await client.get(
                url,
                params={"q": city, "appid": OPENWEATHER_API_KEY, "units": "metric"},
            )
            if resp.status_code == 200:
                data = resp.json()
                name = data.get("name", city)
                country = data.get("sys", {}).get("country", "")
                temp = data["main"]["temp"]
                desc = data["weather"][0]["description"]
                humidity = data["main"]["humidity"]
                logger.info("get_weather success for %s: %s, %s°C", name, desc, temp)
                return f"Current weather in {name}, {country}: {desc}, {temp}°C, humidity {humidity}%."
            return f"Could not find weather data for '{city}'."
    except Exception as e:
        logger.exception("get_weather error for '%s': %s", city, e)
        return f"Unable to retrieve weather for {city} due to a network error."


@tool
@observe(name="get_news")
async def get_news(query: str) -> str:
    """Get the latest news headlines and recent events for a topic, person, company, or location."""
    logger.info("Executing get_news tool for query: '%s'", query)
    try:
        # 1. Primary news search via DuckDuckGoSearchRun (source="news")
        res = await ddg_news_tool.ainvoke(query)
        if res and "No good DuckDuckGo Search Result was found" not in res:
            logger.info("get_news success via news search for: '%s'", query)
            return res
    except Exception as e:
        logger.info("get_news news endpoint error: %s; falling back to text search", e)

    # 2. Fallback to general web search
    try:
        res = await ddg_text_tool.ainvoke(f"{query} news")
        if not res or "No good DuckDuckGo Search Result was found" in res:
            res = await ddg_text_tool.ainvoke(query)
        logger.info("get_news success via web search fallback for: '%s'", query)
        return res or f"No recent news found for '{query}'."
    except Exception as e:
        logger.warning("get_news text search error for '%s': %s", query, e)
        return f"Unable to fetch news at the moment: {e}"


agent_tools = [get_weather, get_news]

# Conversational LLM bound to tools
llm = ChatGroq(
    model=GROQ_MODEL,
    temperature=0.2,
    max_tokens=250,
    reasoning_effort="low",
)
llm_with_tools = llm.bind_tools(agent_tools)


# ==============================================================================
# 3. LangGraph Tool-Calling Agent Workflow
# ==============================================================================

class AgentState(TypedDict):
    """LangGraph conversation state."""
    messages: Annotated[list[BaseMessage], add_messages]


@observe(name="langgraph_agent_node")
async def agent_node(state: AgentState) -> dict:
    """Agent node: decides whether to respond directly or invoke tools."""
    system_prompt = (
        "You are a helpful, conversational AI voice assistant. "
        "Your answer will be spoken aloud to the user using text-to-speech. "
        "Keep replies concise, natural, and under 3 sentences. "
        "Do not use markdown formatting, bullet points, asterisks, or tables. "
        "Use get_weather when asked about weather conditions or temperatures. "
        "Use get_news when asked about current events, news, or latest updates."
    )

    history = list(state.get("messages", []))[-6:]
    messages = [SystemMessage(content=system_prompt)] + history
    logger.info("Agent node evaluating %d messages in context", len(messages))

    response = await llm_with_tools.ainvoke(messages)
    if response.tool_calls:
        logger.info("Agent decided to call tools: %s", [tc["name"] for tc in response.tool_calls])
    else:
        logger.info("Agent generating direct spoken reply")

    return {"messages": [response]}


class VoiceGraphWrapper:
    """
    Filters out internal ToolMessages during streaming so LiveKit's TTS
    only speaks assistant conversational responses, never raw tool data.
    """
    def __init__(self, graph):
        self._graph = graph

    def __getattr__(self, name):
        return getattr(self._graph, name)

    async def astream(self, *args, **kwargs):
        async for item in self._graph.astream(*args, **kwargs):
            if isinstance(item, tuple) and len(item) == 2:
                token, meta = item
                # Suppress tool execution outputs from being streamed to TTS
                if meta.get("langgraph_node") == "tools" or type(token).__name__ == "ToolMessage":
                    continue
            yield item


def build_langgraph_workflow():
    """Assemble and compile the tool-calling LangGraph agent workflow."""
    workflow = StateGraph(AgentState)

    # 1. Add agent and tool nodes
    workflow.add_node("agent", agent_node)
    workflow.add_node("tools", ToolNode(agent_tools))

    # 2. Add edges: start at agent, conditionally route to tools, loop back to agent
    workflow.add_edge(START, "agent")
    workflow.add_conditional_edges("agent", tools_condition)
    workflow.add_edge("tools", "agent")

    compiled_graph = workflow.compile()
    return VoiceGraphWrapper(compiled_graph)


# ==============================================================================
# 4. Telemetry & Metrics Tracking
# ==============================================================================

def setup_session_telemetry(session: AgentSession, ctx: JobContext) -> None:
    """Register usage collection and Time to First Audio (TTFA) telemetry handlers."""
    usage_collector = metrics.UsageCollector()
    last_eou_metrics: metrics.EOUMetrics | None = None

    @session.on("metrics_collected")
    def _on_metrics_collected(ev: MetricsCollectedEvent):
        nonlocal last_eou_metrics
        if ev.metrics.type == "eou_metrics":
            last_eou_metrics = ev.metrics
        metrics.log_metrics(ev.metrics)
        usage_collector.collect(ev.metrics)

    async def log_usage_summary():
        logger.info("Session usage summary: %s", usage_collector.get_summary())

    ctx.add_shutdown_callback(log_usage_summary)

    @session.on("agent_state_changed")
    def _on_agent_state_changed(ev: AgentStateChangedEvent):
        logger.info("Agent state transitioned to: %s", ev.new_state)
        if ev.new_state == "speaking" and last_eou_metrics:
            ttfa = time.time() - last_eou_metrics.timestamp
            logger.info("Time to first audio (TTFA): %.3fs", ttfa)


# ==============================================================================
# 5. LiveKit Voice Agent Server Setup
# ==============================================================================

class VoiceAgent(Agent):
    """VoiceAgent configured for conversational speech interaction."""
    def __init__(self):
        super().__init__(
            instructions=(
                "You are a very helpful AI assistant. Please respond in a clear and concise manner. "
                "Keep replies under 3 sentences and voice-friendly."
            ),
        )


server = AgentServer()


@server.rtc_session()
async def EntryPoint(ctx: JobContext):
    """RTC Session entrypoint: mounts LangGraph agent, STT/TTS fallbacks, and BVC audio."""
    logger.info("Initializing LiveKit RTC Session for room: %s", ctx.room.name)

    # Initialize Langfuse OpenTelemetry tracing
    trace_provider = setup_langfuse(
        metadata={
            "langfuse.session.id": ctx.room.name,
        }
    )
    if trace_provider:
        async def flush_langfuse_traces():
            logger.info("Flushing Langfuse traces on session shutdown...")
            trace_provider.force_flush()

        ctx.add_shutdown_callback(flush_langfuse_traces)

    # Compile the LangGraph tool-calling agent workflow
    langgraph_workflow = build_langgraph_workflow()

    # Wrap the LangGraph workflow as the LiveKit LLM adapter
    langgraph_llm_adapter = LLMAdapter(graph=langgraph_workflow)

    # Configure session with STT/TTS fallbacks and Silero VAD
    session = AgentSession(
        llm=langgraph_llm_adapter,
        stt=stt.FallbackAdapter(
            [
                inference.STT.from_model_string("assemblyai/universal-streaming:en"),
                inference.STT.from_model_string("deepgram/nova-3"),
            ]
        ),
        tts=tts.FallbackAdapter(
            [
                inference.TTS.from_model_string("cartesia/sonic-3:9626c31c-bec5-4cca-baa8-f8ba9e84c8bc"),
                inference.TTS.from_model_string("inworld/inworld-tts-1"),
            ]
        ),
        vad=silero.VAD.load(),
        turn_detection=inference.TurnDetector(),
        preemptive_generation=False,
    )

    # Set up session telemetry & TTFA logging
    setup_session_telemetry(session, ctx)

    # Connect to room with Background Voice Cancellation (BVC)
    await session.start(
        agent=VoiceAgent(),
        room=ctx.room,
        room_options=room_io.RoomOptions(
            audio_input=room_io.AudioInputOptions(
                noise_cancellation=noise_cancellation.BVC(),
            ),
        ),
    )
    logger.info("VoiceAgent started successfully in room: %s", ctx.room.name)

    # Greet the user when they join the session
    await session.generate_reply(
        instructions="Greet the caller warmly in one or two sentences and let them know you can help with weather, news, or any general question."
    )


# ==============================================================================
# 6. CLI Execution
# ==============================================================================

if __name__ == "__main__":
    logging.basicConfig(
        level=logging.INFO,
        format="%(asctime)s [%(levelname)s] %(name)s: %(message)s",
    )
    logger.info("Starting LiveKit Voice Agent Server with LangGraph Tool-Calling Agent...")
    agents.cli.run_app(server)
