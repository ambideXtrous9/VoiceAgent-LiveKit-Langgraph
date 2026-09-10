import logging
from dotenv import load_dotenv
from livekit import agents
from livekit.agents import AgentSession, Agent, AgentServer, JobContext, inference, room_io, TurnHandlingOptions
from livekit.plugins import noise_cancellation, silero 
# Import the multilingual turn detection model
from livekit.plugins.turn_detector.multilingual import MultilingualModel

#Adding fallback adapters
from livekit.agents import llm, stt, tts, inference

# Capturing usage metrics
from livekit.agents import AgentStateChangedEvent, MetricsCollectedEvent, metrics
import time

from telemetry_langfuse import setup_langfuse

logger = logging.getLogger(__name__)

load_dotenv()


class VoiceAgent(Agent):
    def __init__(self):
        super().__init__(
            # Define personality: tone, style, and constraints
            instructions=("You are a very helpful AI assistant. Please respond in a clear and concise manner."
                          "Help the caller fix issues without rambling, and keep replies under 3 sentences."
                        ),

        )

server = AgentServer()

@server.rtc_session()
async def EntryPoint(ctx: JobContext):
    # Initialize Langfuse OpenTelemetry tracing
    trace_provider = setup_langfuse(
        metadata={
            "langfuse.session.id": ctx.room.name,
        }
    )
    if trace_provider:
        async def flush_langfuse_traces():
            logger.info("Flushing Langfuse traces on shutdown...")
            trace_provider.force_flush()

        ctx.add_shutdown_callback(flush_langfuse_traces)

    session = AgentSession(
        # LLM with fallback: OpenAI primary, Gemini backup
        llm=llm.FallbackAdapter(
            [
                inference.LLM(model="openai/gpt-4.1-mini"),
                inference.LLM(model="google/gemini-2.5-flash"),
            ]
        ),
        # STT with fallback: AssemblyAI primary, Deepgram backup
        stt=stt.FallbackAdapter(
            [
                inference.STT.from_model_string("assemblyai/universal-streaming:en"),
                inference.STT.from_model_string("deepgram/nova-3"),
            ]
        ),
        # TTS with fallback: Cartesia primary, Inworld backup
        tts=tts.FallbackAdapter(
            [
                inference.TTS.from_model_string("cartesia/sonic-3:9626c31c-bec5-4cca-baa8-f8ba9e84c8bc"),
                inference.TTS.from_model_string("inworld/inworld-tts-1"),
            ]
        ),
        vad=silero.VAD.load(),                    # Voice activity detection
        turn_detection=MultilingualModel(),       # Semantic turn detection
        preemptive_generation=True,
    )

    # -------------add metrices here---------
    # Aggregate data across all conversation turns
    usage_collector = metrics.UsageCollector()

    # Track End of Utterance timing (when turn detector decides user finished speaking)
    last_eou_metrics: metrics.EOUMetrics | None = None

    @session.on("metrics_collected")
    def _on_metrics_collected(ev: MetricsCollectedEvent):
        nonlocal last_eou_metrics
        # Capture EOU metrics for TTFA calculation
        if ev.metrics.type == "eou_metrics":
            last_eou_metrics = ev.metrics

        # Log each metric as it arrives and add to usage collector
        metrics.log_metrics(ev.metrics)
        usage_collector.collect(ev.metrics)


    async def log_usage():
        # Print per-session summary (tokens, audio duration, costs)
        summary = usage_collector.get_summary()
        logger.info("Usage summary: %s", summary)

    
    # Fire log_usage when worker shuts down
    ctx.add_shutdown_callback(log_usage)

    #Tracking time to first audio
    @session.on("agent_state_changed")
    def _on_agent_state_changed(ev: AgentStateChangedEvent):
        if ev.new_state == "speaking":
            if last_eou_metrics:
                # Calculate time since user finished speaking
                elapsed = time.time() - last_eou_metrics.timestamp
                logger.info(f"Time to first audio: {elapsed:.3f}s")

        
    # Start the session with noise cancellation enabled
    await session.start(
        agent=VoiceAgent(),
        room=ctx.room,
        room_options=room_io.RoomOptions(
            audio_input=room_io.AudioInputOptions(
                noise_cancellation=noise_cancellation.BVC(),  # Background voice cancellation
            ),
        ),
    )


if __name__ == "__main__":
    logging.basicConfig(level=logging.INFO)
    agents.cli.run_app(server)
    