# 🎙️ LiveKit + LangGraph Tool-Calling Voice Agent

A real-time, low-latency conversational voice assistant integrating **LiveKit WebRTC transport** with an autonomous **LangGraph Tool-Calling Agent**. 

Powered by **Groq (`openai/gpt-oss-20b`)**, the agent autonomously decides whether to answer general queries directly or execute real-time tool calls (`get_weather` and `get_news`). Responses are token-streamed in real time directly to LiveKit's text-to-speech (TTS) pipeline, paired with a custom aesthetic reactive voice interface.

---

## 📑 Table of Contents

- [🏛️ System Architecture](#️-system-architecture)
  - [Workflow Diagram](#workflow-diagram)
  - [Audio Pipeline & Multi-Provider Fallbacks](#audio-pipeline--multi-provider-fallbacks)
  - [LangGraph Agent Node & Tool Calling](#langgraph-agent-node--tool-calling)
  - [Real-time Token Streaming Bridge (`VoiceGraphWrapper`)](#real-time-token-streaming-bridge-voicegraphwrapper)
- [🛠️ Tool Specifications](#️-tool-specifications)
- [🎨 Custom Aesthetic Voice Web UI](#-custom-aesthetic-voice-web-ui)
- [📊 Key Metrics & Telemetry](#-key-metrics--telemetry)
- [📋 Prerequisites & API Keys](#-prerequisites--api-keys)
- [📦 Dependencies & Installation](#-dependencies--installation)
- [⚙️ Environment Configuration](#️-environment-configuration)
- [🚀 Execution Guide](#-execution-guide)
  - [1. Development Mode (LiveKit Room Worker)](#1-development-mode-livekit-room-worker)
  - [2. Custom Web UI (Aesthetic Reactive Interface)](#2-custom-web-ui-aesthetic-reactive-interface)
  - [3. Terminal Console Mode (No Browser Needed)](#3-terminal-console-mode-no-browser-needed)
  - [4. Official LiveKit Agent Playground](#4-official-livekit-agent-playground)
  - [5. Production Mode](#5-production-mode)
- [🧪 Automated Testing & Diagnostics](#-automated-testing--diagnostics)
- [💡 Engineering Deep-Dive & Troubleshooting](#-engineering-deep-dive--troubleshooting)
- [📁 Repository Structure](#-repository-structure)
- [📚 References & Further Reading](#-references--further-reading)

---

## 🏛️ System Architecture

### Workflow Diagram

```
                              🎤 User Spoken Audio
                                       │
                                       ▼
                         ┌───────────────────────────┐
                         │   LiveKit WebRTC Room     │
                         │ (Low-latency bidirectional)│
                         └─────────────┬─────────────┘
                                       │
                                       ▼
                         ┌───────────────────────────┐
                         │   LiveKit Voice Pipeline  │
                         │ ├─ Noise Cancellation(BVC)│
                         │ ├─ Silero VAD             │
                         │ ├─ TurnDetector           │
                         │ └─ STT: AssemblyAI        │
                         │       (fallback: Deepgram)│
                         └─────────────┬─────────────┘
                                       │ (Transcribed user speech)
                                       ▼
                         ┌───────────────────────────┐
                         │      LangGraph Agent      │◄────────────────┐
                         │   (Groq: gpt-oss-20b)     │                 │
                         │   bound with agent_tools  │                 │
                         └─────────────┬─────────────┘                 │
                                       │                               │
                      [Tool Call Required? (tools_condition)]          │
                             │                  │                      │
                            YES                 NO                     │
                             │                  │                      │
                             ▼                  │                      │
                      ┌───────────────┐         │                      │
                      │   ToolNode    │         │                      │
                      │ ├─ get_weather│─────────┘                      │
                      │ └─ get_news   │    (ToolMessage execution)     │
                      └───────────────┘                                │
                                                │                      │
                                                ▼                      │
                                     ┌─────────────────────┐           │
                                     │  VoiceGraphWrapper  │           │
                                     │(Filters ToolMessage,│           │
                                     │ yields spoken text) │           │
                                     └──────────┬──────────┘           │
                                                │ (Spoken token stream)│
                                                ▼                      │
                                     ┌─────────────────────┐           │
                                     │  LiveKit Agent TTS  │           │
                                     │ ├─ Primary: Cartesia│           │
                                     │ └─ Backup: Inworld  │           │
                                     └──────────┬──────────┘           │
                                                │                      │
                                                ▼                      │
                                      🔊 Spoken Audio Out              │
```

---

### Audio Pipeline & Multi-Provider Fallbacks

1. **Noise Cancellation & Turn Detection**:
   - **Background Voice Cancellation (`noise_cancellation.BVC`)**: Filters background chatter, ambient noise, and acoustic interference.
   - **Silero Voice Activity Detection (`VAD`)**: Fast boundary detection for voice presence.
   - **Semantic Turn Detector (`inference.TurnDetector`)**: Evaluates natural pauses and end-of-utterance boundaries to prevent cutting off the user prematurely.
2. **Speech-to-Text (STT) Fallback Adapter**:
   - **Primary**: `assemblyai/universal-streaming:en` (streaming transcription with sub-second latency).
   - **Backup**: `deepgram/nova-3` (automatic seamless failover if AssemblyAI encounters network disruptions).
3. **Text-to-Speech (TTS) Fallback Adapter**:
   - **Primary**: `cartesia/sonic-3` (ultra-low TTFB, expressive human-like speech synthesis).
   - **Backup**: `inworld/inworld-tts-1` (automatic failover if Cartesia is unreachable).

---

### LangGraph Agent Node & Tool Calling

The agent uses LangGraph's dynamic ReAct pattern:
- **`AgentState`**: Standard typed message state (`messages: Annotated[list[BaseMessage], add_messages]`).
- **`agent_node`**: Formulates a voice-optimized system prompt (concise, natural, under 3 sentences, no markdown formatting or tables). History is windowed to the 6 most recent messages to maintain low token consumption and avoid context bloat.
- **`llm.bind_tools(agent_tools)`**: Binds the Groq `openai/gpt-oss-20b` reasoning model directly to `@tool` definitions.
- **`tools_condition`**:
  - If tool calls exist, routes to `ToolNode` to execute tools, appending `ToolMessage` outputs into state and looping back to `agent_node` for answer synthesis.
  - If no tool calls exist, routes directly to `END`, streaming tokens to the caller.

---

### Real-time Token Streaming Bridge (`VoiceGraphWrapper`)

When LangGraph executes in `stream_mode="messages"`, it emits messages from **all nodes** in the graph, including raw `ToolMessage` outputs from `ToolNode`. 

Without isolation, LiveKit's `LLMAdapter` would treat raw tool data (e.g. JSON strings, article summaries) as spoken text chunks, causing the text-to-speech engine to speak raw tool data aloud before the assistant synthesizes the answer.

[`langgraph-livekit/agent.py`](langgraph-livekit/agent.py) solves this cleanly with **`VoiceGraphWrapper`**:
```python
class VoiceGraphWrapper:
    """Filters out internal ToolMessages during streaming so LiveKit TTS
    speaks only assistant conversational replies."""
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
```
This guarantees that **only spoken conversational tokens** reach the audio synthesis pipeline.

---

## 🛠️ Tool Specifications

### 1. Weather Tool (`get_weather`)
- **Decorator**: `@tool`
- **Underlying Service**: OpenWeather Current Weather 2.5 API (`https://api.openweathermap.org/data/2.5/weather`)
- **Direct Parameter**: `q={city}` (e.g., `London`, `Tokyo`, `Durgapur`)
- **Units**: Metric (°C)
- **Features**: Single-request execution (~200ms) retrieving location name, country code, weather condition description, temperature, and relative humidity.

### 2. News Tool (`get_news`)
- **Decorator**: `@tool`
- **Underlying Service**: LangChain Community's [`DuckDuckGoSearchRun`](https://reference.langchain.com/python/langchain-community/tools/ddg_search/tool/DuckDuckGoSearchRun)
- **Multi-Source Resiliency**:
  - **Primary**: `DuckDuckGoSearchRun(api_wrapper=DuckDuckGoSearchAPIWrapper(max_results=3, source="news"))` queries real-time news articles.
  - **Fallback**: Automatically falls back to general web search (`source="text"`) if the news endpoint yields rate limits (HTTP 403) or zero results.
- **Output**: Clean article snippets tailored for concise voice summarization.

---

## 🎨 Custom Aesthetic Voice Web UI

A standalone, zero-build web interface is included in [`langgraph-livekit/ui/`](langgraph-livekit/ui/):

- **Mesmerizing Glowing Reactive Voice Orb**: Built with HTML5 Canvas, rendering multi-layered harmonic sine waves that react dynamically to microphone input and incoming agent audio.
- **Glassmorphic Floating Controls**: Start/End Call button, live Mute/Unmute toggle, and active session status pill.
- **Live Subtitles & Conversation Transcript**: Expandable drawer showing the conversation turns and tool execution statuses.
- **Interactive Suggestion Pills**: One-click quick prompts (*"London Weather"*, *"OpenAI Headlines"*, *"Tokyo Forecast"*, *"Fun Fact"*).
- **Zero-Build WebRTC**: Connects directly via LiveKit Client CDN bundle; requires no Node.js or `npm` build step.

---

## 📊 Key Metrics & Telemetry

The agent includes built-in telemetry registered in [`setup_session_telemetry()`](langgraph-livekit/agent.py):

| Metric | What It Measures | Why It Matters |
| :--- | :--- | :--- |
| **TTFA (Time to First Audio)** | Delay from user End of Utterance (EOU) to the first audible synthesized speech packet. | The primary latency metric felt directly by callers (~0.6s–1.0s). |
| **TTFT (Time to First Token)** | Delay until the LLM generates its first completion token. | Isolates LLM latency from STT and TTS synthesis time. |
| **Token Usage** | Cumulative prompt, completion, and reasoning tokens. | Tracked via `metrics.UsageCollector` for cost and rate limit budgeting. |
| **Interruption Rate** | How often the user speaks over the agent. | Indicates conversational pacing and natural turn-taking quality. |
| **Fallback Activations** | Failovers between primary (AssemblyAI/Cartesia) and backup providers. | Measures service reliability and uptime. |

---

## 📋 Prerequisites & API Keys

Ensure credentials are configured for the following services:

| Provider | Purpose | Where to Obtain |
| :--- | :--- | :--- |
| **LiveKit Cloud** | WebRTC media transport & STT/TTS routing | [cloud.livekit.io](https://cloud.livekit.io) |
| **Groq Cloud** | Ultra-fast LLM inference (`openai/gpt-oss-20b`) | [console.groq.com](https://console.groq.com) |
| **OpenWeather** | Real-time global weather data | [openweathermap.org/api](https://openweathermap.org/api) |
| **DuckDuckGo** | Free real-time web & news search | *No API key required* |

---

## 📦 Dependencies & Installation

This project requires **Python 3.13+** and is managed using [`uv`](https://docs.astral.sh/uv/).

### Core Packages (`pyproject.toml`)

```toml
dependencies = [
    "livekit-agents[silero,turn-detector]~=1.3",
    "livekit-plugins-langchain>=1.8.0",
    "livekit-plugins-noise-cancellation~=0.2",
    "langgraph>=1.2.11",
    "langchain-groq>=1.1.3",
    "langchain-community>=0.4.2",
    "duckduckgo-search>=8.1.1",
    "httpx>=0.28.0",
    "python-dotenv>=1.2.3",
]
```

### Installation

From the project root:

```bash
# Sync all dependencies into virtual environment
uv sync

# Download model assets (Silero VAD, Turn Detector)
uv run langgraph-livekit/agent.py download-files
```

---

## ⚙️ Environment Configuration

Copy the example file and add your credentials:

```bash
cp .env.example .env
```

Edit `.env`:

```ini
# LiveKit Cloud Credentials
LIVEKIT_URL=wss://your-project.livekit.cloud
LIVEKIT_API_KEY=devkey_...
LIVEKIT_API_SECRET=secret_...

# Groq Cloud API Key
GROQ_API_KEY=gsk_...

# OpenWeather API Key
OPENWEATHER_API_KEY=your_openweather_api_key
```

---

## 🚀 Execution Guide

### 1. Development Mode (LiveKit Room Worker)
Starts the voice agent worker in dev mode. It connects to your LiveKit Cloud room and waits for participants to join:

```bash
cd langgraph-livekit
uv run agent.py dev
```

### 2. Custom Web UI (Aesthetic Reactive Interface)
To run the full voice experience with the custom reactive browser UI:

```bash
# Terminal 1: Run the agent worker
cd langgraph-livekit && uv run agent.py dev

# Terminal 2: Run the Web UI server
cd langgraph-livekit && uv run python ui/server.py
```
Open **`http://localhost:7860`** in your browser, click the green call button, and speak with your agent!

### 3. Terminal Console Mode (No Browser Needed)
To test and interact with the agent directly through your local computer microphone and speakers:

```bash
cd langgraph-livekit
uv run agent.py console
```

### 4. Official LiveKit Agent Playground
1. Start the agent worker: `cd langgraph-livekit && uv run agent.py dev`
2. Open [LiveKit Agents Playground](https://agents-playground.livekit.io)
3. Enter your `LIVEKIT_URL`, `LIVEKIT_API_KEY`, and `LIVEKIT_API_SECRET`
4. Click **Connect** to start talking.

### 5. Production Mode
To run the agent as a resilient background worker:

```bash
cd langgraph-livekit
uv run agent.py start
```

---

## 🧪 Automated Testing & Diagnostics

A standalone test suite [`langgraph-livekit/test_tools.py`](langgraph-livekit/test_tools.py) is included to verify all tools and workflows end-to-end without needing an active WebRTC session:

```bash
cd langgraph-livekit
uv run python test_tools.py
```

### What the Test Suite Verifies:
1. **Direct Weather Tool (`get_weather`)**: Confirms OpenWeather API returns live temperature, weather conditions, and humidity.
2. **Direct News Tool (`get_news`)**: Confirms `DuckDuckGoSearchRun` executes and extracts article headlines.
3. **End-to-End Weather Tool-Calling Flow**: Sends `"What is the weather in London right now?"` and verifies that the LLM invokes `get_weather` and returns a voice-ready response.
4. **End-to-End News Tool-Calling Flow**: Sends `"What are the top news headlines about OpenAI today?"` and verifies that the LLM invokes `get_news` and summarizes the results.
5. **Direct General Chat Flow**: Sends `"Hello! How are you doing today?"` and verifies that the agent responds immediately without triggering unnecessary tool calls.

---

## 💡 Engineering Deep-Dive & Troubleshooting

### Why `openai/gpt-oss-20b` with `reasoning_effort="low"`?
- The Groq-hosted `openai/gpt-oss-20b` reasoning model provides exceptional tool-calling precision.
- Setting `reasoning_effort="low"` prevents excessive reasoning tokens from delaying speech synthesis, maintaining sub-second Time to First Token (TTFT).
- Setting `max_tokens=250` ensures concise conversational voice replies and avoids hitting Groq's output-tokens-per-minute (OTPM) rate limits.

### Why `preemptive_generation=False`?
- By default, LiveKit can attempt speculative LLM generation while the caller is still finishing a sentence.
- For tool-enabled agents, setting `preemptive_generation=False` is essential to prevent aborted tool calls and duplicate HTTP requests while the user is speaking.

### Resolving DuckDuckGo 403 / Rate-Limit Errors
- DuckDuckGo's `/news.js` endpoint can periodically return HTTP 403 for automated scrapers or non-standard queries.
- `get_news` handles this gracefully: if `ddg_news_tool` encounters an exception or returns empty, it automatically triggers `ddg_text_tool` web search with `f"{query} news"`.

---

## 📁 Repository Structure

```
livekit-voice-agent/
├── .env.example                # Sanitized environment credentials template
├── .gitignore                  # Comprehensive gitignore rules
├── .python-version             # Python version pin (3.13)
├── pyproject.toml              # Dependencies & project metadata
├── uv.lock                     # Locked dependency tree
├── README.md                   # Consolidated project documentation
├── livekit-agent.py            # Original LiveKit reference agent
├── image.png                   # Turn-detection visual reference
└── langgraph-livekit/
    ├── agent.py                # Main LangGraph tool-calling voice agent
    ├── test_tools.py           # Automated diagnostic test suite
    └── ui/
        ├── server.py           # Lightweight token-issuing web server
        └── index.html          # Aesthetic reactive voice web UI
```

---

## 📚 References & Further Reading

- [Worksh.app LiveKit Voice Agent Tutorial: Introduction](https://worksh.app/tutorials/livekit-voice-agent/introduction)
- [Worksh.app LiveKit Voice Agent: Semantic Turn Detection](https://worksh.app/tutorials/livekit-voice-agent/semantic-turn-detection)
- [LiveKit Agents Documentation](https://docs.livekit.io/agents/)
- [LiveKit LangChain Plugin Guide](https://docs.livekit.io/agents/models/llm/langchain/)
- [LangChain DuckDuckGo Search Tool](https://reference.langchain.com/python/langchain-community/tools/ddg_search/tool/DuckDuckGoSearchRun)
- [LangGraph Documentation](https://langchain-ai.github.io/langgraph/)
