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
- [🌐 Frontend & Backend Communication Architecture (FE ⇄ BE)](#-frontend--backend-communication-architecture-fe--be)
  - [Core Communication Paradigm (Decentralized WebRTC SFU)](#core-communication-paradigm-decentralized-webrtc-sfu)
  - [End-to-End Sequence Diagram](#end-to-end-sequence-diagram)
  - [Six-Phase Communication Lifecycle](#six-phase-communication-lifecycle)
  - [Protocol & Data Payload Matrix](#protocol--data-payload-matrix)
  - [Real-Time Interruption & Barge-In Handling](#real-time-interruption--barge-in-handling)
  - [Architecture Deep-Dive: LiveKit WebRTC vs. Conventional FastAPI](#-architecture-deep-dive-livekit-webrtc-vs-conventional-fastapi)
- [🔍 Observability & Tracing with Langfuse](#-observability--tracing-with-langfuse)
  - [How Langfuse Integrates with LiveKit](#how-langfuse-integrates-with-livekit)
  - [Trace Telemetry Breakdown](#trace-telemetry-breakdown)
  - [Langfuse Cloud Dashboard](#langfuse-cloud-dashboard)
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
                                                │                      │
        ═════════════════════════════════════════╪══════════════════════╪══════════════════════════════
                                                │ OpenTelemetry Span Processor
                                                ▼
                                     ┌─────────────────────┐
                                     │ 🔭 Langfuse Tracing │
                                     │ ├─ Session Id (Room)│
                                     │ ├─ STT/TTS Latency  │
                                     │ ├─ Token Usage/Costs│
                                     │ └─ Tool Observations│
                                     └─────────────────────┘
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

## 🌐 Frontend & Backend Communication Architecture (FE ⇄ BE)

A fundamental architectural principle of this system is that **the Frontend (Web UI) does NOT make direct REST API calls, polling requests, or custom WebSocket connections to the Python LangGraph agent**. 

Instead, the browser frontend and backend agent operate as **decoupled WebRTC peers** orchestrated via a centralized **LiveKit WebRTC Selective Forwarding Unit (SFU)**. The only direct HTTP communication is a lightweight initial authentication handshake with the UI server to obtain an ephemeral cryptographic JWT access token.

---

### Core Communication Paradigm (Decentralized WebRTC SFU)

```
┌──────────────────────────────────────────────────────────────────────────────────┐
│                           1. Initial Auth Handshake (HTTP)                       │
│  [Browser FE] ─────────────────── GET /api/token ───────────────► [ui/server.py] │
│  [Browser FE] ◄─────────────── Signed JWT & Room Config ───────── [ui/server.py] │
└──────────────────────────────────────────────────────────────────────────────────┘
                                          │
                                          ▼
┌──────────────────────────────────────────────────────────────────────────────────┐
│                   2. Real-Time WebRTC Media & Signaling Mesh                     │
│                                                                                  │
│    ┌─────────────────┐       WebRTC (WSS / DTLS-SRTP)       ┌────────────────┐   │
│    │                 │ ◄──────────────────────────────────► │                │   │
│    │   Browser FE    │   • Upstream Mic Audio (Opus RTP)    │  LiveKit SFU   │   │
│    │  (index.html)   │   • Downstream Agent Audio Track     │   Cloud Room   │   │
│    │                 │   • Room Events (Speaker Changes)    │                │   │
│    └─────────────────┘                                      └───────┬────────┘   │
│                                                                     │            │
│                                                   WebRTC Sub/Pub    │ RTP Audio  │
│                                                   Bi-directional    │ & Control  │
│                                                                     ▼            │
│                                                             ┌────────────────┐   │
│                                                             │ Agent Worker   │   │
│                                                             │  (agent.py)    │   │
│                                                             │ ├─ VAD / STT   │   │
│                                                             │ ├─ LangGraph   │   │
│                                                             │ └─ TTS Stream  │   │
│                                                             └────────────────┘   │
└──────────────────────────────────────────────────────────────────────────────────┘
```

The system is composed of four distinct layers:
1. **Frontend Client ([`langgraph-livekit/ui/index.html`](langgraph-livekit/ui/index.html))**: A zero-build browser client using the official `livekit-client` JS SDK, Web Audio API, and an HTML5 Canvas reactive audio visualizer.
2. **Token Server ([`langgraph-livekit/ui/server.py`](langgraph-livekit/ui/server.py))**: An asynchronous `aiohttp` web server that mints short-lived, cryptographically signed LiveKit JWT access tokens with granular room permissions.
3. **LiveKit Cloud / SFU (WebRTC Media Router)**: An ultra-low-latency selective forwarding unit that handles ICE/SDP signaling over WebSockets and routes encrypted Opus audio packets over UDP/SRTP between room participants.
4. **Voice Agent Worker ([`langgraph-livekit/agent.py`](langgraph-livekit/agent.py))**: A Python process running the LiveKit Agents SDK and LangGraph. It subscribes to user audio tracks, performs voice activity detection (VAD), runs streaming Speech-to-Text (STT), orchestrates the LangGraph ReAct agent loop (with tool execution), and synthesizes streaming audio back to the room.

---

### End-to-End Sequence Diagram

The following sequence diagram details the full lifecycle from the initial button click in the browser to spoken response playback and canvas visualizer rendering:

```mermaid
sequenceDiagram
    autonumber
    actor User as User / Browser
    participant FE as Web Client (index.html)
    participant Auth as Token Server (ui/server.py)
    participant SFU as LiveKit SFU (WebRTC Cloud)
    participant BE as Voice Agent Worker (agent.py)
    participant Graph as LangGraph Engine (agent_node)
    participant Ext as Cloud Services (Groq / Weather / Cartesia)

    Note over User,Auth: Phase 1: Authentication & Room Allocation
    User->>FE: Clicks Call button (toggleCall())
    FE->>Auth: HTTP GET /api/token?room=&name=
    Auth-->>FE: HTTP 200 { token: "<JWT>", url: "wss://...", room: "voice-...", identity: "caller-..." }

    Note over FE,BE: Phase 2: WebRTC Signaling & Room Connection
    FE->>SFU: room.connect(url, token) via WebSocket
    SFU-->>FE: RoomEvent.Connected
    SFU->>BE: Dispatches session job for room to AgentServer
    BE->>SFU: session.start(agent=VoiceAgent(), room=ctx.room)
    BE->>SFU: session.generate_reply("Greet the caller warmly...")
    SFU-->>FE: Plays initial agent greeting audio

    Note over FE,BE: Phase 3: Upstream Audio Ingestion (FE -> BE)
    FE->>FE: navigator.mediaDevices.getUserMedia({ audio: true })
    FE->>FE: Attach mic to Web Audio Analyser (FFT frequency capture)
    FE->>SFU: room.localParticipant.setMicrophoneEnabled(true) [Opus 48kHz RTP]
    SFU->>BE: Relays user audio stream
    Note over BE: BVC noise cancellation -> Silero VAD -> TurnDetector (EOU)
    BE->>Ext: AssemblyAI streaming STT
    Ext-->>BE: Transcribed user text: "What's the weather in Tokyo?"

    Note over BE,Graph: Phase 4: LangGraph Agentic Reasoning & Tool Calling
    BE->>Graph: Injects query as HumanMessage into AgentState
    Graph->>Ext: Groq (openai/gpt-oss-20b) reasoning
    alt Tool Required
        Ext-->>Graph: Tool Call: get_weather("Tokyo")
        Graph->>Ext: OpenWeather 2.5 API request
        Ext-->>Graph: ToolMessage("Tokyo, JP: Clear, 18°C")
        Graph->>Ext: Groq synthesizes conversational voice reply
    end
    Note over Graph,BE: VoiceGraphWrapper suppresses ToolMessage; streams clean assistant tokens

    Note over BE,FE: Phase 5: Downstream Audio Synthesis & Delivery (BE -> FE)
    BE->>Ext: Streams spoken text tokens to Cartesia Sonic-3 TTS
    Ext-->>BE: Synthesized PCM audio packets
    BE->>SFU: Publishes Agent Audio Track into room
    SFU-->>FE: RoomEvent.TrackSubscribed (kind: audio)
    FE->>FE: track.attach() appends <audio id="agentAudio"> to DOM (Instant playback)

    Note over FE,BE: Phase 6: Reactive State Synchronization & Canvas Visualizer
    SFU-->>FE: RoomEvent.ActiveSpeakersChanged([speaker])
    alt Speaker is Local User
        FE->>FE: Set state = 'listening' (Cyan/Emerald aura)
        FE->>FE: Canvas renderOrb() reads micAnalyser byte frequency data
    else Speaker is Remote Agent
        FE->>FE: Set state = 'speaking' (Magenta/Purple aura)
        FE->>FE: Canvas renderOrb() reads agentAnalyser byte frequency data
    else Silence
        FE->>FE: Set state = 'idle' (Soft ambient aura)
    end
```

---

### Six-Phase Communication Lifecycle

#### Phase 1: Authentication & Room Allocation (HTTP Handshake)
When the user clicks the call button in [`langgraph-livekit/ui/index.html`](langgraph-livekit/ui/index.html), the frontend initiates an HTTP `GET /api/token` request to the token server ([`langgraph-livekit/ui/server.py`](langgraph-livekit/ui/server.py)):

```javascript
// index.html
const resp = await fetch('/api/token');
const { token, url, room: roomName } = await resp.json();
```

Inside [`ui/server.py`](langgraph-livekit/ui/server.py), the `handle_token()` handler:
1. Generates an ephemeral room name (e.g. `voice-a1b2c3`) and unique participant identity (`caller-d4e5`).
2. Constructs a signed LiveKit JWT using `LIVEKIT_API_KEY` and `LIVEKIT_API_SECRET`.
3. Attaches granular video/audio grants:
   ```python
   token = (
       api.AccessToken(api_key=LIVEKIT_API_KEY, api_secret=LIVEKIT_API_SECRET)
       .with_identity(identity)
       .with_name(participant_name)
       .with_grants(
           api.VideoGrants(
               room_join=True,
               room=room_name,
               can_publish=True,
               can_subscribe=True,
               can_publish_data=True,
           )
       )
   )
   jwt_token = token.to_jwt()
   ```
4. Returns the connection bundle: `{ token, url, room, identity, name }`.

#### Phase 2: WebRTC Signaling & Room Enrollment
The browser initializes a LiveKit room instance using the official JavaScript client SDK:

```javascript
room = new LivekitClient.Room({
  adaptiveStream: true,
  dynacast: true,
});
await room.connect(url, token);
```

- **Signaling Channel**: The client connects to `LIVEKIT_URL` over a secure WebSocket (`wss://`).
- **WebRTC Peer Connection**: The browser and LiveKit SFU exchange SDP offers/answers and ICE candidates, establishing an encrypted DTLS-SRTP media channel over UDP.
- **Agent Assignment**: LiveKit Cloud assigns the newly created room to the running Python agent worker (`agent.py`) via its persistent `AgentServer` connection. The agent executes `EntryPoint(ctx: JobContext)` and joins the exact same room `ctx.room`.

#### Phase 3: Upstream Audio Ingestion (Frontend ➔ Agent Worker)
Once connected, the browser acquires the user's microphone stream and publishes it to the LiveKit room:

```javascript
micStream = await navigator.mediaDevices.getUserMedia({ audio: true });
await room.localParticipant.setMicrophoneEnabled(true);
```

- **Audio Encoding**: Browser encodes mic input into an Opus audio track (48 kHz sample rate) and streams it as RTP packets to the LiveKit SFU.
- **LiveKit Voice Pipeline Processing** ([`langgraph-livekit/agent.py`](langgraph-livekit/agent.py)):
  1. **Background Voice Cancellation (`noise_cancellation.BVC`)**: Removes background chatter, fans, and ambient noise.
  2. **Silero VAD**: Analyzes audio frames in 30ms windows to detect speech presence.
  3. **Turn Detection (`inference.TurnDetector`)**: Evaluates natural speech cadence and determines the End of Utterance (EOU) timestamp.
  4. **Streaming STT Fallback Adapter**: Forwards raw audio to **AssemblyAI** (`assemblyai/universal-streaming:en`), falling back seamlessly to **Deepgram** (`deepgram/nova-3`) if the primary stream experiences latency spikes or errors. The transcribed text string is emitted as soon as the user finishes their sentence.

#### Phase 4: Agentic Reasoning & Tool Isolation (`VoiceGraphWrapper`)
The transcribed text is packaged into a `HumanMessage` and injected into the LangGraph state machine:

1. **State Machine (`agent_node`)**: Evaluates conversation history and queries **Groq (`openai/gpt-oss-20b`)** with low reasoning effort.
2. **Autonomous Tool Routing**:
   - If user asks about the weather, `tools_condition` routes to `ToolNode(agent_tools)` to execute `get_weather(city)`.
   - If user asks about current events, `tools_condition` routes to `get_news(query)`.
   - The tool output (`ToolMessage`) is fed back into `agent_node` to formulate a concise, spoken conversational response.
3. **Streaming Isolation Bridge ([`VoiceGraphWrapper`](langgraph-livekit/agent.py))**:
   - Standard LangGraph streaming outputs chunks from *every* node in the graph, including raw tool outputs.
   - `VoiceGraphWrapper` intercepts `graph.astream()` and drops all `ToolMessage` instances and tokens originating from the `"tools"` node:
     ```python
     if meta.get("langgraph_node") == "tools" or type(token).__name__ == "ToolMessage":
         continue
     yield item
     ```
   - This ensures **only clean conversational assistant tokens** reach the audio synthesis adapter.

#### Phase 5: Downstream Audio Synthesis & Playback (Agent Worker ➔ Frontend)
1. **Text-to-Speech Synthesis**: Filtered assistant tokens are streamed directly to LiveKit's `tts.FallbackAdapter` (primary **Cartesia Sonic-3**, fallback **Inworld TTS**). Cartesia begins streaming PCM audio chunks within ~150ms of receiving the first token.
2. **Audio Track Publication**: The agent worker publishes the synthesized audio stream into the WebRTC room as a remote audio track.
3. **Frontend Playback ([`langgraph-livekit/ui/index.html`](langgraph-livekit/ui/index.html))**:
   The browser listens for the incoming audio track and attaches it directly to an HTML `<audio>` tag:
   ```javascript
   room.on(LivekitClient.RoomEvent.TrackSubscribed, (track, publication, participant) => {
     if (track.kind === LivekitClient.Track.Kind.Audio) {
       const el = track.attach();
       el.id = 'agentAudio';
       document.body.appendChild(el);
     }
   });
   ```
   The browser immediately renders and plays the agent's voice with minimal buffering.

#### Phase 6: Reactive State Synchronization & Canvas Visualizer
The UI maintains perfect visual sync with the conversation without polling:

1. **Speaker State Detection**: LiveKit automatically detects active speakers on both audio tracks and fires `RoomEvent.ActiveSpeakersChanged`:
   ```javascript
   room.on(LivekitClient.RoomEvent.ActiveSpeakersChanged, (speakers) => {
     if (speakers.length > 0) {
       const speaker = speakers[0];
       if (speaker === room.localParticipant) {
         currentAgentState = 'listening';
         orbStatus.textContent = 'Listening to you...';
       } else {
         currentAgentState = 'speaking';
         orbStatus.textContent = 'Aura Speaking';
       }
     } else {
       currentAgentState = 'idle';
       orbStatus.textContent = 'Aura Listening';
     }
   });
   ```
2. **Web Audio API Frequency Analysis**:
   - The frontend routes both the user's mic stream and the agent's incoming audio track through Web Audio `AnalyserNode` instances (`micAnalyser` and `agentAnalyser`).
   - The HTML5 Canvas animation loop (`renderOrb()`) reads the Fast Fourier Transform (FFT) frequency byte data (`getByteFrequencyData()`) 60 times per second.
   - Dynamic sine-wave radius, color gradients (Cyan for listening, Magenta for speaking, Indigo for idle), and harmonic wave oscillations react proportionally to vocal volume in real time.

---

### Protocol & Data Payload Matrix

| Stage | Channel / Connection | Sender | Receiver | Protocol | Payload / Content |
| :--- | :--- | :--- | :--- | :--- | :--- |
| **Token Request** | HTTP GET `/api/token` | Browser FE | `ui/server.py` | HTTP/1.1 (JSON) | `{ "room": "...", "name": "..." }` |
| **Token Response** | HTTP 200 OK | `ui/server.py` | Browser FE | HTTP/1.1 (JSON) | `{ "token": "<JWT>", "url": "wss://...", "room": "...", "identity": "..." }` |
| **Room Signaling** | LiveKit Signaling URL | Browser / Agent | LiveKit Cloud | WebSocket (`wss://`) | SDP offers/answers, ICE candidates, participant metadata |
| **User Voice (Up)** | WebRTC Audio Track | Browser FE | LiveKit Cloud SFU | SRTP / Opus (48kHz) | User microphone raw audio packets |
| **User Voice (In)** | WebRTC Audio Track | LiveKit Cloud SFU | `agent.py` Worker | SRTP / Opus (48kHz) | Forwarded user audio packets into Silero VAD & STT |
| **Agent Voice (Out)** | WebRTC Audio Track | `agent.py` Worker | LiveKit Cloud SFU | SRTP / Opus (48kHz) | Cartesia Sonic-3 synthesized audio packets |
| **Agent Voice (Down)**| WebRTC Audio Track | LiveKit Cloud SFU | Browser FE | SRTP / Opus (48kHz) | Subscribed agent audio track attached to `<audio>` element |
| **Speaker Sync** | WebRTC Data/Event Channel| LiveKit Cloud SFU | Browser FE | WebRTC Protobuf | `RoomEvent.ActiveSpeakersChanged` (Active speaker identities) |
| **Audio Visualization**| Local In-Memory | Web Audio API | HTML5 Canvas | Native Web Audio API | `Uint8Array` FFT frequency data mapped to sine-wave contours |

---

### Real-Time Interruption & Barge-In Handling

One of the key benefits of this WebRTC architecture over traditional HTTP/WebSocket chat interfaces is **instant conversational barge-in**:

1. **User Speaks Over Agent**: When the agent is speaking (playing Cartesia TTS audio through the browser), the user can begin speaking at any time.
2. **Instant VAD Detection**: Silero VAD on the agent worker detects user voice activity while the agent is in the `speaking` state.
3. **Immediate Track Flush**:
   - The LiveKit agent worker immediately cancels the pending TTS synthesis stream.
   - LiveKit flushes the queued audio packets from the WebRTC track buffer.
   - The agent transitions to listening mode and starts a new STT transcription turn.
4. **Instant UI Reaction**:
   - LiveKit SFU broadcasts an `ActiveSpeakersChanged` event indicating the local participant is now active.
   - The frontend instantly stops the agent audio visualizer, switches the orb gradient from magenta to cyan, and updates the status label to *"Listening to you..."* with zero audible echo or delay.

---

## ⚖️ Architecture Deep-Dive: LiveKit WebRTC vs. Conventional FastAPI

A common architectural question when building voice AI applications is: **"Why not just build this with a conventional FastAPI backend using WebSockets or REST endpoints?"**

Understanding the answer requires dissecting how the `langgraph-livekit` backend actually communicates with the frontend, the physical network protocols involved, and the severe limitations of standard HTTP/TCP stacks for bi-directional human speech.

---

### Architectural Topologies Compared

#### Topology A: Conventional FastAPI Architecture (Direct Monolithic Client-Server over TCP)

```
┌──────────────────────────────────────────────────────────────────────────────────┐
│                         CONVENTIONAL FASTAPI ARCHITECTURE                        │
└──────────────────────────────────────────────────────────────────────────────────┘

 [ Browser / Client ]                                    [ Python FastAPI Server ]
 ┌──────────────────┐                                    ┌───────────────────────┐
 │                  │         HTTP POST /api/chat        │                       │
 │  MediaRecorder   │ ─────────────────────────────────> │  • Python Asyncio Loop│
 │  (Blobs: 2-5s)   │   Audio File (.wav / .webm)        │    (Under GIL load)   │
 │                  │                                    │  • STT: Whisper       │
 │                  │         HTTP 200 / SSE Audio       │  • LLM: LangChain     │
 │  AudioContext    │ <───────────────────────────────── │  • TTS: ElevenLabs    │
 │  (Wait & Play)   │      Chunked MP3 / WAV stream      │                       │
 └──────────────────┘                                    └───────────────────────┘
          │                                                          │
          │                       OR via WebSocket                   │
          │             ws://api.domain.com/ws/audio (TCP)           │
          └──────────────────────────────────────────────────────────┘
                  ▲                                          ▲
                  │  🔴 High Latency: 2,500ms – 6,000ms      │
                  │  🔴 Head-of-Line Blocking on packet loss │
                  │  🔴 Fragile manual interruption logic    │
                  │  🔴 Python CPU handles raw audio packets │
                  └──────────────────────────────────────────┘
```

#### Topology B: LiveKit Decoupled WebRTC Architecture (SFU-Mediated Mesh over UDP)

```
┌───────────────────────────────────────────────────────────────────────────────────────────┐
│                     LIVEKIT + LANGGRAPH DECOUPLED ARCHITECTURE (OUR REPO)                 │
└───────────────────────────────────────────────────────────────────────────────────────────┘

 [ Web Frontend (Browser) ]                                  [ LiveKit SFU Media Server ]
 ┌──────────────────────────┐                                ┌──────────────────────────┐
 │ • LiveKit Client SDK     │                                │ • High-Performance Go/C++│
 │ • WebRTC AudioContext    │    1. HTTP Token Request       │ • Low-Latency SFU Engine │
 │ • Hardware AEC / AGC     │ ─────────────────────────────> │ • STUN / TURN NAT Relay  │
 │ • Audio Visualizer Canvas│                                │ • Global Edge Network    │
 └─────────────┬────────────┘                                └─────────────┬────────────┘
               │                                                           │
               │         2. WebRTC PeerConnection (UDP / SRTP / Opus)      │
               ├───────────────────────────────────────────────────────────┤
               │   • Upstream: User Voice Track (48kHz Opus @ 32kbps)      │
               │   • Downstream: Agent Spoken Track (Instant playback)     │
               │   • Data Channel: Transcripts, State, Events (<20ms)      │
               │                                                           │
               │                                             ▲             │
               │                                             │             │
               │                     3. WebRTC PeerConnection│(Local/Mesh) │
               │                     (UDP / SRTP / Opus)     ▼             │
               │                                     ┌─────────────────────┴────┐
               │                                     │ LangGraph Agent Worker   │
               │                                     │ (`langgraph-livekit/`)   │
               │                                     ├──────────────────────────┤
               │                                     │ • Silero VAD (Zero-delay)│
               │                                     │ • TurnDetector (Semantics│
               │                                     │ • AssemblyAI STT         │
               │                                     │ • LangGraph State Graph  │
               │                                     │   (Groq gpt-oss-20b)     │
               │                                     │ • Cartesia Sonic TTS     │
               │                                     └──────────────────────────┘
```

---

### Step-by-Step: How the Frontend Communicates with the Agent Backend

Unlike traditional web applications where the browser connects directly to a Python HTTP or WebSocket port:

1. **Authentication Handshake (Only HTTP Step)**:
   - When the user opens the web app or clicks **"Connect to Agent"**, the frontend sends a single `POST /api/token` request to the lightweight token server ([`ui/server.py`](ui/server.py) on port `8080`).
   - The token server mints an HMAC-SHA256 signed **LiveKit JWT** embedding room permissions (`roomJoin`, `canPublish: true`, `canSubscribe: true`, participant identity) and returns it along with the LiveKit WebSocket/WebRTC URL (`LIVEKIT_URL`).

2. **WebRTC PeerConnection Establishment**:
   - The frontend instantiates `new LivekitClient.Room()`.
   - It performs an SDP (Session Description Protocol) offer/answer exchange with the **LiveKit SFU** over a signaling WebSocket, then establishes direct peer-to-peer media paths using **ICE (Interactive Connectivity Establishment)** via UDP (falling back to STUN/TURN if behind symmetric NAT/firewalls).

3. **Autonomous Agent Worker Registration**:
   - In parallel, the Python agent backend ([`agent.py`](langgraph-livekit/agent.py)) runs as an independent daemon using the `livekit-agents` worker runtime.
   - It connects to the same LiveKit SFU server over a secure worker control channel.
   - When the user enters the room, the LiveKit SFU dispatches a job event to the worker pool. The agent worker accepts the job and joins the exact same room as an active participant named `agent`.

4. **Continuous Bi-Directional Audio Streaming (Full-Duplex UDP)**:
   - **Frontend $\rightarrow$ Agent**: The browser publishes a local WebRTC audio track (`MediaStreamTrack`). The microphone stream is encoded into **Opus packets (20ms frames at 48kHz)** and forwarded by the LiveKit SFU over UDP to the Python agent worker.
   - **Agent $\rightarrow$ Frontend**: When the agent speaks, Cartesia streams raw PCM audio chunks $\rightarrow$ LiveKit Agent encodes them into Opus $\rightarrow$ publishes an agent audio track $\rightarrow$ LiveKit SFU forwards RTP packets directly to the browser $\rightarrow$ browser renders audio through the Web Audio API.

5. **Out-of-Band Real-Time Telemetry & State Synchronization**:
   - Transcripts, participant speaking states, tool execution notifications, and telemetry metadata travel over the **WebRTC Data Channel** (SCTP over DTLS).
   - This provides sub-20ms event synchronization without polluting the audio stream or polling HTTP endpoints.

---

### Why Not Conventional FastAPI? The 8 Critical Failure Modes

| Dimension | Conventional FastAPI (HTTP / WebSocket) | LiveKit WebRTC Architecture (This Repo) | Why It Matters for Voice AI |
| :--- | :--- | :--- | :--- |
| **Transport Layer** | **TCP** (HTTP/1.1, HTTP/2, WebSockets) | **UDP / SRTP** (WebRTC Media Plane) | **Head-of-Line Blocking**: TCP guarantees delivery by retransmitting lost packets. On real-world Wi-Fi/4G with 1-2% packet loss, TCP halts the stream for 400–1200ms. In voice, late audio is useless; WebRTC drops lost packets and uses Opus PLC. |
| **Turn-Taking Latency** | **3,000ms – 6,000ms** (Half-duplex audio recording blobs) | **300ms – 650ms** (Continuous streaming full-duplex) | Human conversation feels unnatural if turn-taking latency exceeds 700ms. FastAPI blob uploading destroys conversational cadence. |
| **Barge-In / Interruption** | **Extremely difficult & fragile** (Requires client detection, custom WS cancel commands, task aborts) | **Native & Instantaneous (<100ms)** (Silero VAD at agent worker flushes SFU track buffer) | If the user speaks while the bot is talking, FastAPI bots keep playing stale audio for 1-2 seconds. LiveKit cuts off mid-syllable the millisecond user voice energy is sensed. |
| **Acoustic Echo Cancellation (AEC)** | **Prone to feedback loops** (Browser mic picks up speaker output; bot listens to itself) | **Native WebRTC AEC & AGC** hardware pipeline integration | Without WebRTC's echo canceller tied to the output audio device, the bot's own voice triggers its STT, causing infinite conversational loops. |
| **Server Event Loop Load** | **Python GIL bottleneck** (FastAPI event loop parses WS frames, audio chunks, and TLS) | **Zero media processing in Python** (Go/C++ SFU handles packet switching & routing) | FastAPI Python processes freeze under concurrent audio encoding/decoding. In LiveKit, Python only handles AI orchestration; high-throughput audio routing is handled by the SFU. |
| **Network Traversal (NAT/Firewalls)** | Standard HTTP ports (80/443), but fails on restrictive P2P or real-time streaming | Built-in **STUN, TURN, ICE, & UDP multiplexing** | Ensures 99.99% connection success across corporate firewalls, mobile carrier NATs, and VPNs. |
| **Multi-Party Scalability** | Requires building custom room management, Redis Pub/Sub, and audio mixing | **Native Room Architecture** out of the box | Seamlessly supports adding human listeners, screen-sharing, supervisor monitoring, or multi-agent collaboration in the same room. |
| **Client Audio Player Management** | Manual Web Audio API buffering, jitter buffer, and audio node scheduling | **Managed LiveKit Client SDK** with automatic adaptive jitter buffer | Prevents audio underruns, clicks, pops, and drift without writing hundreds of lines of fragile JavaScript audio pipeline code. |

---

### Detailed Analysis of Core Technical Differences

#### 1. The TCP "Head-of-Line" Blocking Problem vs. UDP Media Streaming

A standard FastAPI WebSocket implementation relies on **TCP (Transmission Control Protocol)**.
- **In TCP**: Every packet must be acknowledged. If Packet #14 is dropped over a mobile connection, Packets #15, #16, and #17 **cannot be delivered to the application** until Packet #14 is retransmitted and acknowledged.
- **The Result in Voice**: The browser audio playback freezes, waits 600ms, and then plays back an accelerated "chipmunk" burst of delayed audio.
- **WebRTC's Solution**: LiveKit uses **SRTP (Secure Real-Time Transport Protocol) over UDP**. In conversational speech, a 20ms audio frame from 500ms ago is completely worthless. WebRTC discards late packets, and the **Opus Packet Loss Concealment (PLC)** algorithm mathematically interpolates the missing sound seamlessly. Latency never builds up.

#### 2. Half-Duplex Batching vs. Streaming Full-Duplex

In a typical FastAPI voice bot:
```
User clicks Mic ──> MediaRecorder records 3 sec ──> User clicks Stop ──> POST .wav (2MB) ──> FastAPI saves to disk ──> Whisper transcribes ──> LLM generates ──> TTS generates .mp3 ──> Response sent ──> Browser plays audio
Total Turnaround: 4,000ms – 7,000ms
```

In this LiveKit + LangGraph repository:
```
User speaks ──> 20ms Opus frames stream over UDP ──> Silero VAD detects speech boundary (<50ms) ──> AssemblyAI streaming STT produces interim tokens ──> Groq LLM streams first token in 180ms ──> Cartesia Sonic synthesizes audio stream in 100ms ──> WebRTC RTP audio plays in browser
Total Turnaround: ~450ms – 650ms (Human-level conversational speed)
```

#### 3. Why Barge-In is Broken in FastAPI WebSockets

In FastAPI, canceling an in-flight audio playback requires a brittle daisy-chain:
1. The frontend's JavaScript needs its own VAD library running in WebAssembly (consuming high CPU on mobile devices).
2. Upon sensing speech, the frontend sends a JSON WebSocket message: `{"action": "interrupt"}`.
3. The FastAPI server must intercept this JSON message while its Python `asyncio` worker is concurrently streaming TTS audio chunks.
4. Python must cancel the active generator task, notify the TTS provider to stop charging tokens, and send an acknowledgment.
5. The browser must empty its local Web Audio buffer.
During this multi-step roundtrip (typically **800ms – 1,500ms**), the bot continues blaring audio from the user's speakers, speaking right over the user.

In LiveKit:
- Silero VAD runs **directly inside the LiveKit Agent worker process** receiving the continuous audio stream.
- When user speech energy crosses the threshold during an agent speaking turn, the agent's internal `AgentSession` immediately:
  - Aborts the Cartesia HTTP chunk stream.
  - Flushes the local WebRTC track publisher queue.
  - Tells the LiveKit SFU to send a `trackMuted` or silence packet.
  - Fires an `ActiveSpeakersChanged` event.
- Total latency: **under 100 milliseconds**. The bot halts speaking almost instantaneously.

---

### Code Comparison: FastAPI vs. LiveKit Agent

#### The FastAPI Approach (Hundreds of lines of fragile state machine code)

```python
# CONVENTIONAL FASTAPI (FRAGILE & COMPLEX)
from fastapi import FastAPI, WebSocket, WebSocketDisconnect
import asyncio

app = FastAPI()

@app.websocket("/ws/voice")
async def voice_endpoint(websocket: WebSocket):
    await websocket.accept()
    audio_buffer = bytearray()
    playback_task = None
    is_speaking = False

    try:
        while True:
            # Under TCP, audio packets and control packets share the same queue
            data = await websocket.receive()
            if "bytes" in data:
                audio_buffer.extend(data["bytes"])
                # Must manually detect silence, manage chunking, handle jitter
            elif "text" in data and data["text"] == "STOP":
                # Manual cancellation of audio playback
                if playback_task and not playback_task.done():
                    playback_task.cancel()
                    await websocket.send_json({"status": "cancelled"})
    except WebSocketDisconnect:
        pass
```

#### The LiveKit Approach (Clean, declarative, and production-hardened)

```python
# OUR ARCHITECTURE (LIVEKIT AGENTS + LANGGRAPH)
from livekit.agents import JobContext, WorkerOptions, cli
from livekit.agents.voice import VoicePipelineAgent

async def entrypoint(ctx: JobContext):
    # Connect directly to the WebRTC room managed by LiveKit SFU
    await ctx.connect()

    # Native VAD, STT, and TTS with built-in WebRTC track publishing and barge-in
    agent = VoicePipelineAgent(
        vad=silero.VAD.load(),
        stt=assemblyai.STT(),
        llm=VoiceGraphWrapper(langgraph_runnable),  # LangGraph State Machine
        tts=cartesia.TTS(),
        chat_ctx=initial_ctx,
    )

    # Start conversational agent with automatic track routing & echo cancellation
    agent.start(ctx.room)
    await agent.say("Hello! How can I assist you today?")

if __name__ == "__main__":
    cli.run_app(WorkerOptions(entrypoint_fnc=entrypoint))
```

---

### Summary: The Architectural Verdict

- **Use FastAPI When**: You are building REST APIs, CRUD microservices, database backends, or text-based chat applications where requests are discrete and latency tolerances are in the hundreds of milliseconds or seconds.
- **Use LiveKit WebRTC When**: You are building **conversational voice AI** that requires human-speed response times (<700ms), robust acoustic echo cancellation, instant voice interruption (barge-in), resilient performance over lossy wireless networks (UDP), and multi-party room infrastructure.

---

## 🔍 Observability & Tracing with Langfuse

This project features native, production-grade observability powered by **[Langfuse](https://langfuse.com)** and **OpenTelemetry**. Every conversation, speech recognition turn, tool execution, and audio synthesis event is automatically captured and streamed to your Langfuse dashboard.

### How Langfuse Integrates with LiveKit

LiveKit Agents includes built-in OpenTelemetry support. Our integration ([`telemetry_langfuse.py`](langgraph-livekit/telemetry_langfuse.py)) registers Langfuse as an OpenTelemetry span processor:

```python
from langfuse import Langfuse
from opentelemetry.sdk.trace import TracerProvider
from livekit.agents.telemetry import set_tracer_provider

def setup_langfuse(metadata=None):
    trace_provider = TracerProvider()
    set_tracer_provider(trace_provider, metadata=metadata)
    Langfuse(
        public_key=os.getenv("LANGFUSE_PUBLIC_KEY"),
        secret_key=os.getenv("LANGFUSE_SECRET_KEY"),
        base_url=os.getenv("LANGFUSE_BASE_URL"),
        tracer_provider=trace_provider,
        should_export_span=lambda span: True,
    )
    return trace_provider
```

### Trace Telemetry Breakdown

1. **Session & Room Correlation**: Each LiveKit room session is mapped directly to `langfuse.session.id = ctx.room.name`, allowing you to inspect full multi-turn conversations grouped by room or caller ID.
2. **Speech-to-Text (STT) Spans**: Captures transcription latency, active audio stream intervals, and fallback switches (AssemblyAI ➔ Deepgram).
3. **LangGraph Agent & Tool Execution**:
   - The conversational agent node is decorated with `@observe(name="langgraph_agent_node")`.
   - Tool calls (`get_weather`, `get_news`) are instrumented with `@observe`, capturing precise arguments (`city`, `query`), execution latency, and return values.
4. **Text-to-Speech (TTS) Spans**: Tracks synthesis time, TTFA (Time to First Audio), token consumption, and fallback audio providers (Cartesia ➔ Inworld).
5. **Zero Data Loss on Shutdown**: Registers `trace_provider.force_flush()` via `ctx.add_shutdown_callback` to ensure all pending spans are securely transmitted when participants disconnect or the worker shuts down.

### Langfuse Cloud Dashboard

Navigate to [Langfuse Cloud](https://cloud.langfuse.com) (or [US Region](https://us.cloud.langfuse.com)) to inspect:
- **Trace Waterfall**: Millisecond-accurate breakdown of VAD, STT, LLM reasoning, tool calls, and TTS synthesis.
- **Token & Cost Analytics**: Cumulative prompt, completion, and reasoning tokens across sessions.
- **Latency Breakdown**: Compare TTFT (Time to First Token) vs TTFA (Time to First Audio) to isolate network and synthesis bottlenecks.

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
| **Langfuse Cloud** | Production voice agent tracing & observability | [cloud.langfuse.com](https://cloud.langfuse.com) |
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
    "langfuse>=2.0.0",
    "opentelemetry-sdk>=1.25.0",
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

# Langfuse Observability & Tracing
LANGFUSE_PUBLIC_KEY=pk-lf-...
LANGFUSE_SECRET_KEY=sk-lf-...
LANGFUSE_BASE_URL=https://us.cloud.langfuse.com # Or https://cloud.langfuse.com (EU)

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

The project includes a comprehensive suite of automated tests to verify system functionality and resilience across all edge cases without needing an active browser session:

### 1. Comprehensive Edge-Case Test Suite (`test_edgecases.py`)
Verifies full system resilience across edge cases and error conditions:

```bash
uv run python langgraph-livekit/test_edgecases.py
```

**What it validates:**
- **Weather Tool Edge Cases**: Standard cities (`Tokyo`), multi-word locations (`San Francisco`), accented Unicode cities (`São Paulo`), empty/whitespace strings (graceful prompts), and non-existent cities (404 handled gracefully).
- **News Tool Edge Cases**: Standard queries (`Artificial Intelligence`), special characters/symbols (`NVIDIA & AMD @ 2026!?`), empty/whitespace queries, and automatic fallback search.
- **Workflow & Streaming**: 10-message context windowing, `@observe` span creation, and strict `VoiceGraphWrapper` suppression preventing `ToolMessage` chunks from leaking into the audio stream.
- **Langfuse Configuration & Safe Flush**: Tests `is_langfuse_configured()`, empty-key graceful fallback, `flush_langfuse(None)` safe no-op, and active nested trace flush.
- **UI Server & Token Generation**: Route registration (`/`, `/health`, `/api/token`), JWT signing, and room grants.

### 2. Langfuse Observability Test Suite (`test_langfuse.py`)
Verifies your Langfuse credentials, OpenTelemetry tracer provider registration, synthetic voice pipeline spans, and live cloud export:

```bash
uv run python langgraph-livekit/test_langfuse.py
```

### 3. LangGraph Agent & Tool Calling Suite (`test_tools.py`)
Verifies both Weather and News tools independently and within the end-to-end LangGraph tool-calling agent workflow:

```bash
uv run python langgraph-livekit/test_tools.py
```

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

### Langfuse Observability & Debug Mode
If traces are not appearing in your Langfuse dashboard:
1. **Enable Debug Mode**:
   ```bash
   export LANGFUSE_DEBUG="True"
   ```
2. **Check Base URL / Data Region**:
   - 🇺🇸 US Cloud: `LANGFUSE_BASE_URL="https://us.cloud.langfuse.com"`
   - 🇪🇺 EU Cloud: `LANGFUSE_BASE_URL="https://cloud.langfuse.com"`
3. **Verify Flush Callbacks**: Ensure `ctx.add_shutdown_callback(trace_provider.force_flush)` is registered to export buffered spans before worker process termination.

---

## 📁 Repository Structure

```
livekit-voice-agent/
├── .env.example                # Sanitized environment credentials template (includes Langfuse)
├── .gitignore                  # Comprehensive gitignore rules
├── .python-version             # Python version pin (3.13)
├── pyproject.toml              # Dependencies & project metadata (includes langfuse & opentelemetry-sdk)
├── uv.lock                     # Locked dependency tree
├── README.md                   # Consolidated project documentation
├── livekit-agent.py            # Reference LiveKit agent with fallback adapters & Langfuse
├── telemetry_langfuse.py       # Root Langfuse OpenTelemetry setup helper
├── image.png                   # Turn-detection visual reference
└── langgraph-livekit/
    ├── agent.py                # Main LangGraph tool-calling voice agent with Langfuse tracing
    ├── telemetry_langfuse.py   # Langfuse OpenTelemetry configuration module
    ├── test_edgecases.py       # Comprehensive edge-case & resilience test suite
    ├── test_langfuse.py        # Automated Langfuse connectivity & tracing verification
    ├── test_tools.py           # Automated diagnostic test suite
    └── ui/
        ├── server.py           # Lightweight token-issuing web server (with /health)
        └── index.html          # Aesthetic reactive voice web UI
```

---

## 📚 References & Further Reading

- [Langfuse LiveKit Integration Guide](https://langfuse.com/integrations/frameworks/livekit)
- [Langfuse Observability & Tracing Documentation](https://langfuse.com/docs)
- [Worksh.app LiveKit Voice Agent Tutorial: Introduction](https://worksh.app/tutorials/livekit-voice-agent/introduction)
- [Worksh.app LiveKit Voice Agent: Semantic Turn Detection](https://worksh.app/tutorials/livekit-voice-agent/semantic-turn-detection)
- [LiveKit Agents Documentation](https://docs.livekit.io/agents/)
- [LiveKit LangChain Plugin Guide](https://docs.livekit.io/agents/models/llm/langchain/)
- [LangChain DuckDuckGo Search Tool](https://reference.langchain.com/python/langchain-community/tools/ddg_search/tool/DuckDuckGoSearchRun)
- [LangGraph Documentation](https://langchain-ai.github.io/langgraph/)

