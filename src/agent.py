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

from src.graph import VoiceGraphWrapper, build_langgraph_workflow
from src.telemetry import flush_langfuse, setup_langfuse

load_dotenv(find_dotenv())

logger = logging.getLogger("livekit.langgraph_agent")


def setup_session_telemetry(session: AgentSession, ctx: JobContext) -> None:
    """Register usage collection and Time to First Audio (TTFA) telemetry handlers."""
    usage_collector = metrics.UsageCollector()
    last_eou_metrics: metrics.EOUMetrics | None = None

    @session.on("metrics_collected")
    def _on_metrics_collected(ev: MetricsCollectedEvent):
        nonlocal last_eou_metrics
        try:
            if ev.metrics.type == "eou_metrics":
                last_eou_metrics = ev.metrics
            metrics.log_metrics(ev.metrics)
            usage_collector.collect(ev.metrics)
        except Exception as e:
            logger.warning("Error processing collected metric: %s", e)

    async def log_usage_summary():
        try:
            logger.info("Session usage summary: %s", usage_collector.get_summary())
        except Exception as e:
            logger.warning("Failed to retrieve usage summary: %s", e)

    ctx.add_shutdown_callback(log_usage_summary)

    @session.on("agent_state_changed")
    def _on_agent_state_changed(ev: AgentStateChangedEvent):
        try:
            logger.info("Agent state transitioned to: %s", ev.new_state)
            if ev.new_state == "speaking" and last_eou_metrics:
                ttfa = time.time() - last_eou_metrics.timestamp
                logger.info("Time to first audio (TTFA): %.3fs", ttfa)
        except Exception as e:
            logger.warning("Error processing agent state change: %s", e)


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
            flush_langfuse(trace_provider)

        ctx.add_shutdown_callback(flush_langfuse_traces)

    # Compile the LangGraph tool-calling agent workflow wrapped for voice
    compiled_graph = build_langgraph_workflow()
    langgraph_workflow = VoiceGraphWrapper(compiled_graph)

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


if __name__ == "__main__":
    logging.basicConfig(
        level=logging.INFO,
        format="%(asctime)s [%(levelname)s] %(name)s: %(message)s",
    )
    logger.info("Starting LiveKit Voice Agent Server with LangGraph Tool-Calling Agent...")
    agents.cli.run_app(server)
