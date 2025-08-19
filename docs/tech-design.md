# Mythic — Technical Design

## 0) Goals & Non-Goals

**Goals**

* Real-time conversational NPCs with speech in/out.
* One shared engine serving many characters concurrently on a single machine (preferably 12–24 GB VRAM) or a small server.
* Tight latency: end-of-user-utterance → first audible reply **≤ 800 ms** (streaming).
* Hot-swappable character personas, lore, and episodic memory via lightweight RAG.
* Long-term: centralized orchestrator coordinating remote ASR/LLM/TTS/RAG workers per component.

**Non-Goals**

* Not doing gigantic models per character.
* Not doing deep end-to-end emotion recognition, gesture synthesis, or world-model LLM planning (stub hooks only).

---

## 1) End-to-End Pipeline

**Audio In → VAD → ASR (streaming) → Dialogue Manager → RAG → LLM → TTS (streaming) → Audio Out**

Latency budget (99p targets):

* **VAD gate**: 0–20 ms per frame
* **ASR partials**: first partial ≤ 150 ms; final text ≤ 150 ms after endpoint
* **DM + RAG**: ≤ 50 ms
* **LLM first token**: 300–700 ms (7–8B @ Q4 on mid-range GPU)
* **TTS first audio chunk**: ≤ 150 ms; then stream 40–80 ms chunks

Initial SLOs (MVP):

* p95 end-to-end ≤ 900 ms; p99 ≤ 1200 ms (utterance_end → first_audio_ms).
* p95 LLM TTFT ≤ 700 ms on 7–8B Q4; revisit with KV paging and scheduler improvements.

---

## 2) Deployment Profiles

* **Local single GPU (preferred MVP)**

  * GPU: 12–24 GB (e.g., 4070 / 4090).
  * LLM 7–8B Q4 shared by all NPCs.
  * ASR/TTS: CPU where possible (or small GPU slice).
* **Remote microservice** (for console/low-end PCs):

  * gRPC/WebSocket to engine; low-latency network <20 ms RTT recommended.

* **Orchestrated microservices (long-term)**

  * Central orchestrator (control-plane) managing remote workers for ASR, LLM, TTS, and Vector/RAG.
  * Data-plane: gRPC streaming for control + WS binary frames for audio; service discovery and health checks.
  * Autoscaling per component; token/KV-aware scheduling; backpressure across queues.

---

## 3) Core Services & Processes

### 3.1 Processes

* **Gateway** (FastAPI/gRPC/WS): routes events to per-NPC actors.
* **ASR Pool**: N workers (CPU-bound), each with VAD + streaming ASR.
* **LLM Server**: 1 instance with multi-session, paged KV, token scheduler.
* **TTS Pool**: M workers (CPU or GPU) with streaming output.
* **Vector/RAG**: in-proc FAISS/Chroma per NPC (tiny indexes); shared embedding encoder.

### 3.2 Concurrency Model

* **Per-NPC Actor** (async task / coroutine) owns:

  * Dialogue state machine
  * Short text history
  * NPC memory handles (persona, lore, episodic, scene)
  * Mailbox (queue) for events: `SpeechStart`, `PartialASR`, `FinalASR`, `LLMReady`, `AudioDone`, `SceneUpdate`, etc.
* **Schedulers**

  * **LLM Token Scheduler**: fair-share + priority weighting.
  * **TTS Worker Pool**: FIFO with preemption by “on-screen speaking” flag.

---

## 4) Data Contracts (JSON over WS/gRPC messages)

### 4.1 Audio → ASR

Production: audio payloads should use binary WebSocket frames or gRPC streaming. JSON examples below are illustrative; include `schema_version` in all control messages.

```json
{
  "schema_version": "v1",
  "type": "AudioChunk",
  "npc_id": "barkeep_01",
  "session_id": "s-124",
  "pcm": "<bytes base64>",
  "sample_rate": 16000,
  "ts_client": 1723909123.102
}
```

**ASR Outputs**

```json
{ "schema_version": "v1", "type": "ASRPartial", "npc_id": "barkeep_01", "text": "did you say", "is_final": false }
{ "schema_version": "v1", "type": "ASRFinal", "npc_id": "barkeep_01", "text": "Did you say the tavern is closed?", "lang": "en" }
```

### 4.2 DM → LLM (Tool Calling JSON)

```json
{
  "schema_version": "v1",
  "type": "LLMRequest",
  "npc_id": "barkeep_01",
  "turn_id": "t-309",
  "system": "<persona + guardrails + scene schema>",
  "messages": [
    {"role":"user","content":"Did you say the tavern is closed?"}
  ],
  "tools_schema": [
    {"name":"give_item","schema":{"type":"object","properties":{"player_id":{"type":"string"},"item_id":{"type":"string"}},"required":["player_id","item_id"]}},
    {"name":"set_waypoint","schema":{"type":"object","properties":{"x":{"type":"number"},"y":{"type":"number"},"z":{"type":"number"}},"required":["x","y","z"]}}
  ],
  "retrieval_hints": {"persona_keys":["tone","backstory"], "lore_tags":["tavern","market_hours"], "scene_scope":"zone_3"},
  "max_tokens": 128,
  "temperature": 0.7,
  "top_p": 0.9
}
```

**LLM Outputs**

```json
{
  "schema_version":"v1",
  "type":"LLMStreamDelta",
  "npc_id":"barkeep_01",
  "turn_id":"t-309",
  "delta":"No, friend—",
  "tool_call": null
}
```

or tool use:

```json
{
  "schema_version":"v1",
  "type":"LLMToolCall",
  "name":"set_waypoint",
  "call_id":"c-88310",
  "arguments":{"x":12.4,"y":0.0,"z":-3.1}
}
```

### 4.3 LLM → TTS

```json
{
  "schema_version":"v1",
  "type":"Synthesize",
  "npc_id":"barkeep_01",
  "voice_id":"baritone_02",
  "text":"No, friend—open till the moon’s high.",
  "streaming": true,
  "viseme": true, 
  "sample_rate": 24000
}
```

**TTS Stream**

```json
{ "schema_version":"v1", "type":"TTSChunk", "npc_id":"barkeep_01", "pcm":"<b64>", "visemes":[{"t":0.12,"p":"AA"}, ...] }
```

### 4.4 Errors & Acks

```json
{ "schema_version":"v1", "type":"Error", "code":"TOOL_TIMEOUT", "message":"set_waypoint timed out", "turn_id":"t-309", "call_id":"c-88310", "ref":"e-72f4" }
{ "schema_version":"v1", "type":"Ack", "ack_of":"ToolCall", "turn_id":"t-309", "call_id":"c-88310", "status":"ok" }
```

---

## 5) Dialogue Manager (DM)

### 5.1 State Machine

States per NPC: `Idle` → `Listening` → `Thinking` → `Speaking` (+ `BargeIn`).
Transitions triggered by `ASRFinal`, `LLMReady`, `AudioDone`.
Backchannels (“mm-hm…”) can be scheduled on partials via a small heuristic.

### 5.2 Intent & Tooling

* **Classifier** (tiny) on final ASR text → intents: `chitchat | trade | quest | hint | combat | navigation | admin`.
* DM chooses which memory channels to enable and which tools to expose.
* All tool actions are **externally executed** (game engine), LLM only emits tool JSON.

### 5.3 Safety & Constraints

* Max tokens/call; max concurrent speaking NPCs.
* Disallow topic classes (config) or swear level per character.
* Style enforcer post-filter (regex/short model) to keep voice consistent.

### 5.4 Speech Chunking / Phrase Buffering

* Buffer ~200–400 ms of LLM text to form stable phrase boundaries before handing to TTS to reduce prosody churn.
* Allow early emission for backchannels or very short replies; otherwise coalesce to 1–2 sentences.
* On barge-in: cancel outstanding TTS segments, mark turn aborted, and deprioritize current session in the LLM scheduler.

---

## 6) RAG & Memory

### 6.1 Memory Channels

* **Persona**: static cards (tone, dialect, backstory, do/don’t). Stored as 3–8 short documents (≤2k tokens total).
* **Lore/World**: quest docs, items, locations with tags; stored per zone; chunk size 256–512 tokens, overlap 64.
* **Episodic**: last N interactions summarized; time-decayed priority.
* **Scene State**: current objects/actors in FOV; fed as a compact JSON.

### 6.2 Retrieval Pipeline (fast path)

1. **Query build** from latest user text + intent + scene tags.
2. **Dual retrieval**:

   * **BM25** (fast lexical) top-K=20
   * **Embeddings** (384-dim) top-K=20
3. **Light rerank** (mini-cross-encoder or cosine blend) → top-K=6.
4. Pack into **context sections** with headers the LLM can follow:

   * `## Persona`, `## Lore`, `## Last Scene`, `## Recent Memory`.

### 6.3 Indexes

* **Per-NPC**: persona (tiny), episodic (tiny).
* **Per-zone**: lore shards (FAISS/Chroma).
* Stored locally: `./memory/{npc_id}/…` and `./memory/zone_{id}/…`

### 6.4 Memory Updates

* After each turn: summarize long exchanges into 1–2 sentences (mini model or prompt).
* TTL/decay: exponential weight by time + frequency of reference.

### 6.5 Context Token Budgets

* Hard cap total context tokens (excluding generated tokens) per request.
* Suggested budgets: `Persona≤200`, `Lore≤400`, `Recent Memory≤200`, `Scene≤150`.
* Enforce truncation and section-level drop policy when over budget (drop least-relevant `Scene` objects first, then trim `Lore`).

---

## 7) Models & Sizing

### 7.1 LLM

* **Size**: 7–8B instruct, **Q4** GGUF (llama.cpp) or TensorRT-LLM INT4.
* **VRAM (rule-of-thumb)**:

  * Weights 7B Q4: \~3.5–4.5 GB.
  * Runtime & buffers: \~1–2 GB.
  * **KV cache per active speaking session**:

    * Approx = `tokens_context * bytes_per_val * 2 * n_layers * n_heads * head_dim / hidden_size_factor` (impl-dependent).
    * In practice (fp16 KV): **\~0.5–1.0 GB** for 2–4k context on 7–8B.
    * With 8-bit/FP8 KV: \~40–60% of fp16.
* **Throughput** (consumer GPU): 25–60 tok/s single stream; 2–8 streams with scheduler.

**Capacity targets**

* **12 GB VRAM** → 2–3 active speakers comfortably.
* **24 GB VRAM** → 5–8 active speakers or a 13B Q4 with fewer speakers.

### 7.2 ASR

* **faster-whisper** small/int8 or base/int8; 16 kHz mono.
* Chunk: 20 ms frames, 200 ms segments; WebRTC VAD to gate.

### 7.3 TTS

* **Piper** for CPU-fast (≤10 ms per 100 ms audio).
* **XTTS-v2** (GPU) for quality; use streaming phoneme API if available.

---

## 8) LLM Scheduling & KV Management

### 8.1 Token Scheduler (pseudocode)

```python
while True:
    ready = [s for s in sessions if s.needs_tokens()]
    ready.sort(key=lambda s: (priority(s), -s.wait_time))
    for s in ready:
        budget = burst_tokens(s)   # e.g., 32–64
        gen = llm.generate_step(s, budget)  # yields tokens
        yield_to_audio_if_speaking(s)
```

**Priority(s)**:

* +2 if `s.on_screen and s.mic_active`
* +1 if `s.speaking`
* 0 default
* -1 if `s.idle > 10s`

**Backpressure & fairness**

* Minimum service slice per idle session to avoid starvation (e.g., 8 tokens per round).
* Apply backpressure to upstream mailboxes when LLM/TTS queues exceed thresholds.

### 8.2 KV Paged Attention

* Use paged KV to share GPU memory; evict oldest **summarized** turns when off-screen.
* **Resume strategy**: keep a compact seed (system + 1-2 recap messages) in KV; reload when NPC re-enters view.

---

## 9) Prompting & Guardrails

### 9.1 System Template (per NPC)

```
You are {name}, a {archetype}. Style: {tone}. 
NEVER break character. Keep replies short (<= 2 sentences) unless the player asks for details.
Follow GAME RULES strictly: {rules...}
You can call tools only via JSON exactly matching provided schemas. If unsure, ask a brief question.
Context sections may follow; only use them if relevant.
```

### 9.2 Context Packing

```
## Persona
- speaks in clipped, sardonic phrases
- won't reveal secret passage outright

## Lore
- Tavern hours: sunset to 02:00
- Guard patrols pass market every 5 min

## Recent Memory
- Player asked about smugglers; you refused details.

## Scene
{"zone":"market","time":"21:10","nearby":["guard","stall","lamp"]}
```

### 9.3 Tool Responses

* Tool responses are appended as assistant-role messages with a short, structured “result” that the LLM must verbalize.

---

## 10) TTS & Lip-Sync

* **Sample rate**: 24 kHz preferred for quality; 16 kHz works for speed.
* **Chunk size**: 40–80 ms per emitted PCM frame.
* **Visemes/phonemes**: request from TTS; fallback to G2P if missing.
* **Barge-in**: if player interrupts, DM sends `Cutoff` → stop TTS stream; LLM scheduler deprioritizes current turn.
* **Phrase buffering**: accumulate ~200–400 ms text before synthesis for smoother prosody; stream 40–80 ms PCM chunks thereafter.

---

## 11) Scene & Tools API (game engine)

### 11.1 Engine→AI (scene updates)

```json
{ "type":"SceneUpdate", "npc_id":"barkeep_01",
  "objects":[{"id":"guard_3","dist":5.1},{"id":"player","dist":1.2}],
  "time":"21:10", "zone":"market" }
```

### 11.2 AI→Engine (tool calls)

```json
{ "schema_version":"v1", "type":"ToolCall", "name":"set_waypoint", "args":{"x":12.4,"y":0.0,"z":-3.1}, "npc_id":"barkeep_01", "turn_id":"t-309", "call_id":"c-88310" }
```

**Minimum tools**: `give_item`, `set_waypoint`, `open_dialog`, `play_emote`, `set_focus(target)`, `set_marker(location)`.

Ack/idempotency: engine acks with `{type:"Ack"...}`; duplicate `call_id` must be safely ignored. Standard timeouts and retries with backoff.

---

## 12) Storage Layout

```
/models/llm/…               # GGUF / engine-specific
/models/tts/…               # voices
/memory/persona/{npc}/…     # yaml cards, tiny vector store
/memory/lore/zone_{id}/…    # shard indexes
/memory/episodic/{npc}/…    # small sqlite/chroma
/logs/…                      # structured logs
```

Persona YAML example:

```yaml
name: Barkeep
tone: dry, helpful, guarded
taboos: [reveal_secret_passage, price_haggling]
catchphrases: ["friend", "listen"]
```

---

## 13) Observability

### 13.1 Metrics

* **ASR**: WER (offline eval), partial latency, finalization latency.
* **LLM**: TTFT, tok/s, queue wait, tokens/turn, KV usage, OOMs.
* **TTS**: time-to-first-chunk, RT-factor (synth\_time / audio\_len).
* **End-to-end**: utterance\_end → first\_audio\_ms.
* **Quality**: style adherence %, lore accuracy probes.

### 13.2 Tracing & Logs

* Correlate by `session_id`, `npc_id`, `turn_id`.
* Store tool calls + arguments + outcomes.

### 13.3 SLOs (initial)

* p95 end-to-end ≤ 900 ms; p99 ≤ 1200 ms (utterance_end → first_audio_ms).
* p95 LLM TTFT ≤ 700 ms on 7–8B Q4; revisit with KV paging.

---

## 14) Config Flags (per NPC & global)

* `max_concurrent_speakers`: default 2
* `llm.max_tokens`: 128
* `llm.temperature`: 0.6–0.9
* `scheduler.burst_tokens`: 32–64
* `kv.max_ctx_tokens`: 1024–2048 (summarize beyond)
* `tts.voice_id`: e.g., `baritone_02`
* `safety.blocklist`: terms/categories
* `schema.version`: v1

---

## 15) Security & Abuse

* Strip/escape tool arguments.
* Length caps per field.
* Optional bad-content filter before TTS.
* Sandboxed tool executor (no direct eval).

---

## 16) Test Plan

### 16.1 Functional

* **Happy path**: Q\&A with tool call (waypoint).
* **Interrupt**: player barges in mid-speech → stop and respond.
* **Off-screen**: NPC leaves view → KV evict & resume seed works.
* **RAG**: modify lore index; NPC obeys new hours.

### 16.2 Load

* 2, 4, 8 NPCs speaking concurrently; measure TTFT, tok/s, audio stutter.
* 60-minute soak: memory growth, file descriptors, GPU mem.

### 16.3 Quality

* Style probes: 20 prompts per NPC; human-rated style adherence.
* Lore correctness: 50 questions with known answers.

---

## 17) Risks & Mitigations

* **KV explosion** → summarize early, 8-bit KV, strict ctx caps.
* **TTS startup pops** → pre-warm voices; emit short silence first chunk.
* **ASR hallucinations** in noise → aggressive VAD thresholds; push-to-talk.
* **Latency spikes** when many talk → scheduler weights + cap simultaneous speakers to 2.

---

## 18) Implementation Notes (sane defaults)

### 18.1 Tech Choices

* **Server**: Python 3.11 + FastAPI + uvicorn (WebSocket) or gRPC.
* **Actors**: `asyncio` tasks; one queue per NPC (`asyncio.Queue`).
* **LLM**: llama.cpp server (`--ctx-size 2048 --parallel N --mlock` if RAM allows), Q4\_K\_M GGUF.
  * Alternative: **vLLM** or **TensorRT-LLM** for stronger multi-session, paged KV, and better TTFT under load.
* **Embeddings**: small 384-dim encoder (fast, CPU OK).
* **Vector DB**: Chroma (duckdb) or FAISS in-proc.
* **ASR**: faster-whisper streaming wrapper + WebRTC VAD (20 ms).
* **TTS**: Piper / XTTS-v2 with streaming.

### 18.2 Skeleton (very abridged)

**NPC Actor**

```python
class NPCAgent:
    def __init__(self, npc_id, q_in, q_out, memory):
        self.state = "Idle"
    async def run(self):
        while True:
            evt = await self.q_in.get()
            if evt.type == "ASRFinal":
                ctx = memory.retrieve(evt.text, scene=evt.scene, intent=classify(evt.text))
                await llm.enqueue(self.npc_id, build_prompt(ctx, evt.text))
                self.state = "Thinking"
            elif evt.type == "LLMStreamDelta":
                await tts.enqueue_stream(self.npc_id, evt.delta)
                self.state = "Speaking"
            elif evt.type == "AudioDone":
                self.state = "Idle"
```

**LLM Scheduler (loop)**

```python
while True:
    s = pick_next_session()      # weighted RR
    tokens = llm.generate(s, burst=48)
    dispatch_tokens(s.npc_id, tokens)
```

---

## 19) Milestones

**M0 (Week 1–2)**

* Single NPC: streaming ASR → LLM → TTS.
* Hard-coded persona; no RAG.
* TTFT < 1.2 s.

**M1 (Week 3–4)**

* 3 NPCs, shared LLM; fair scheduler; cap speakers=2.
* Basic RAG (persona + lore).
* Unity/Unreal WS plugin + viseme lip-sync.

**M2 (Week 5–6)**

* Tool calling to engine APIs.
* Episodic memory + summarization.
* Load test 60 min; TTFT p95 < 900 ms.

**M3 (Week 7+)**

* LoRA personality adapters (optional).
* KV 8-bit, better reranker, polish voices.
* Orchestrator pilot: remote ASR/LLM/TTS workers + token-aware scheduler.

---

## 20) Config Examples

**Engine config (YAML)**

```yaml
schema_version: v1
gpu:
  llm_model: ./models/llm/mistral7b.Q4_K_M.gguf
  ctx_tokens: 2048
  kv_precision: fp8
scheduler:
  burst_tokens: 48
  max_speakers: 2
asr:
  vad: webrtc
  lang: en
tts:
  backend: piper
  sample_rate: 24000
rag:
  embedder: "mini-embed-384"
  topk_bm25: 20
  topk_vec: 20
  topk_final: 6
  budgets:
    persona: 200
    lore: 400
    recent: 200
    scene: 150
safety:
  blocklist: ["credit card", "exact address"]
```

**Persona card**

```yaml
name: "Marla, the Barkeep"
style: "wry, concise, suspicious of strangers"
rules:
  - "Never reveal the smuggler tunnel outright"
  - "Offer hints if the player mentions 'blue lantern'"
voice_id: "baritone_02"
```

---

## 21) Orchestrator Mode (Control-Plane + Remote Workers)

**Control-plane (orchestrator)**

* Maintains session registry, per-NPC actor routing, and global token scheduler.
* Performs service discovery, health checks, and autoscaling hints.
* Implements end-to-end backpressure: pauses inbound audio when downstream is saturated.

**Remote workers (data-plane)**

* ASR workers (CPU-bound), TTS workers (CPU/GPU), LLM server (GPU), Vector/RAG indexers.
* Control channel: gRPC (Protobuf) for requests/acks/errors with `schema_version`.
* Audio channel: WS binary or gRPC streaming; Opus optional inbound, PCM outbound.
* Contracts include `turn_id` and `call_id` for idempotency and tracing.

**Scheduling**

* KV-aware token scheduler prioritizes on-screen/mic-active sessions; minimum slice for idle sessions.
* TTS pool supports preemption for on-screen speakers.

**Operations**

* Deploy via Docker Compose locally; Kubernetes for remote; readiness/liveness and warmup routes.
* Centralized metrics/tracing (OpenTelemetry) with SLO dashboards and queue depth gauges.

