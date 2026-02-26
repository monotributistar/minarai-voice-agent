# Voice Agent MVP (Groq STT + Groq LLM + Groq/API, Piper, or F5 TTS)

This folder adds a local voice-agent stack for the portfolio:

- Frontend (React + Vite): push-to-talk panel over WebSocket
- Orchestrator (Node): audio conversion + STT + LLM + TTS pipeline
- STT:
  - Groq Whisper API (`STT_PROVIDER=groq`) recommended for trial quota tests
  - Optional local fallback (FastAPI + faster-whisper) with Docker profiles
- TTS:
  - Groq API (`TTS_PROVIDER=groq`) for quick POC
  - Local Piper fallback (`TTS_PROVIDER=local`)
  - Local F5 Spanish (`TTS_PROVIDER=f5`) with voice cloning (`ref_audio` + `ref_text`)

## Services and ports

- Orchestrator WS: `ws://localhost:8787`
- STT HTTP: `http://localhost:9001`
- TTS HTTP: `http://localhost:9002`
- F5 TTS HTTP (profile `f5`): `http://localhost:9003`

## Windows 11 setup checklist (Docker Desktop + WSL2 + NVIDIA)

1. Docker Desktop:

- Enable **Use the WSL 2 based engine**.
- Enable WSL integration for your distro.

2. NVIDIA and GPU:

- Install/update NVIDIA Windows drivers.
- Ensure Docker Desktop and WSL2 are updated.
- In Docker Desktop settings, enable GPU support if your version exposes that toggle.

3. Validate GPU inside a container:

```bash
docker run --rm --gpus all nvidia/cuda:12.4.1-base-ubuntu22.04 nvidia-smi
```

If this fails, typical fixes:

- install latest NVIDIA driver,
- confirm WSL2 backend is active,
- restart Docker Desktop,
- update WSL (`wsl --update`).

## Quick start

```bash
cp voice-agent/.env.example voice-agent/.env
```

### Mode A: Groq STT + Groq LLM (fastest MVP path)

Set in `.env`:

- `STT_PROVIDER=groq`
- `TTS_PROVIDER=groq`
- `LLM_BASE_URL=https://api.groq.com/openai/v1`
- `LLM_MODEL=llama-3.1-8b-instant` (or your preferred Groq model)
- `LLM_API_KEY=<your_groq_key>`
- `GROQ_API_KEY=<your_groq_key>` (optional if `LLM_API_KEY` already set)

Groq TTS notes:

- Endpoint: `/audio/speech`
- Current voice model support is mainly English for `playai-tts`.
- For Spanish requests, set `GROQ_TTS_NON_EN_STRATEGY=local` to auto-fallback to Piper.
- `GROQ_TTS_INPUT_MAX_CHARS` defaults to `180` for reliable MVP behavior.

Run:

```bash
cd voice-agent
docker compose up --build
```

### Mode B: local faster-whisper STT (GPU profile, RTX 3070)

```bash
cd voice-agent
docker compose --profile gpu up --build
```

### Mode C: local faster-whisper STT (CPU fallback profile)

```bash
cd voice-agent
docker compose --profile cpu up --build
```

### Mode D: F5 Spanish TTS (voice cloning)

Set in `.env`:

- `TTS_PROVIDER=f5`
- `F5_MODEL_ID=jpgallegoar/F5-Spanish`
- `F5_DEVICE=cuda` (or `cpu` if no GPU)
- `F5_REF_AUDIO_ES=/app/voices/es/reference.wav`
- `F5_REF_TEXT_ES=<transcript of your reference audio>`
- `F5_TTS_FALLBACK_PROVIDER=local` (recommended for resilience)

Prepare your reference voice:

1. Save a clean WAV sample at `voice-agent/tts-f5/voices/es/reference.wav`.
2. Use 8-20 seconds of dry speech (no music/noise), single speaker, consistent mic.
3. Write the exact transcript in `F5_REF_TEXT_ES` (must match audio).

Run:

```bash
cd voice-agent
docker compose --profile f5 up --build
```

In another terminal (repo root):

```bash
pnpm dev
```

## Piper voices (used when `TTS_PROVIDER=local` or fallback)

Put Piper models in `voice-agent/tts/voices` and match paths in `.env`:

- `PIPER_VOICE_ES`
- `PIPER_VOICE_EN`

Example expected layout:

- `voice-agent/tts/voices/es/es_ES-sharvard-medium.onnx`
- `voice-agent/tts/voices/en/en_US-joe-medium.onnx`

You can use your preferred Piper voices; this repo does not bundle model weights.

## F5 model and voice cloning notes

- The F5 service downloads the model from Hugging Face (`F5_MODEL_ID`) on first startup unless `F5_MODEL_LOCAL_PATH` is set.
- Default model: [jpgallegoar/F5-Spanish](https://huggingface.co/jpgallegoar/F5-Spanish).
- You do not need one model per voice. For cloning, you only need a reference sample + exact transcript.
- Cloning quality depends mostly on reference quality and transcript accuracy.
- Keep one reference per target voice/accent; change `F5_REF_AUDIO_ES` + `F5_REF_TEXT_ES` to switch speaker.
- You can override per request by calling `/speak` with `ref_audio` and `ref_text`.

Reference preparation example:

```bash
ffmpeg -i raw_reference.m4a -ac 1 -ar 24000 -c:a pcm_s16le voice-agent/tts-f5/voices/es/reference.wav
```

Then set in `.env`:

- `F5_REF_AUDIO_ES=/app/voices/es/reference.wav`
- `F5_REF_TEXT_ES=...` (exact spoken text in the reference audio)

## WebSocket contract

### Client -> server (two-frame audio upload)

1. Text frame metadata:

```json
{
  "type": "audio_meta",
  "requestId": "<uuid>",
  "lang": "es-AR",
  "mime": "audio/webm;codecs=opus",
  "tts": {
    "voice": "es",
    "style": "rioplatense_friendly",
    "prosody": { "speed": 0.97, "pitch": -0.05, "energy": 0.78, "pause_ms": 130 },
    "generation": { "temperature": 0.6, "stability": 0.78, "seed": 42 },
    "audio": { "format": "wav", "sample_rate": 22050 },
    "constraints": { "max_chars": 340 }
  }
}
```

2. Binary frame with raw recorded bytes.

### Server -> client

Transcripts:

```json
{"type":"transcript","requestId":"<uuid>","role":"user","text":"...","isFinal":true}
{"type":"transcript","requestId":"<uuid>","role":"assistant","text":"...","isFinal":true}
```

Audio:

```json
{ "type": "audio_meta", "requestId": "<uuid>", "mime": "audio/wav" }
```

Then binary frame with WAV bytes.

Optional state updates:

```json
{
  "type": "state",
  "requestId": "<uuid>",
  "state": "uploading|transcribing|thinking|speaking|idle|error"
}
```

Debug ping:

```json
{ "type": "debug_ping", "lang": "es-AR", "tts": { "instance_id": "es_ES-sharvard-medium" } }
```

Server debug status:

```json
{
  "type": "debug_status",
  "data": {
    "ok": true,
    "ttsProvider": "local",
    "ttsService": { "healthy": true },
    "ttsInstances": [{ "id": "es_ES-sharvard-medium", "lang": "es", "modelPath": "...", "source": "configured" }],
    "activeTtsConfig": { "instance_id": "es_ES-sharvard-medium" }
  }
}
```

Provider toggle (active/inactive):

```json
{ "type": "provider_set", "provider": "groq", "enabled": false, "lang": "es-AR" }
```

## Pipeline details

Per request:

1. Receive `audio/webm;codecs=opus` blob from browser.
2. Convert with ffmpeg to WAV PCM 16kHz mono.
3. STT:
   - `STT_PROVIDER=groq`: `POST https://api.groq.com/openai/v1/audio/transcriptions`
   - `STT_PROVIDER=local`: `POST /transcribe` to local STT service
4. TTS:
   - `TTS_PROVIDER=groq`: `POST https://api.groq.com/openai/v1/audio/speech`
   - `TTS_PROVIDER=local`: `POST /speak` to local Piper wrapper
   - `TTS_PROVIDER=f5`: `POST /speak` to local F5 wrapper (`tts-f5`)
5. Build persona prompt + optional CMS grounding snippets.
6. Call external LLM API (OpenAI-compatible chat-completions adapter).
7. Post-process assistant text for short TTS-friendly response.
8. Return WAV via WebSocket.

## TTS node config

The orchestrator accepts `audio_meta.tts` as a structured generation payload.
For node-based providers (`local`, `f5`, `groq`) the orchestrator delegates synthesis and model resolution to the TTS node via a standardized HTTP contract.

- `voice`: logical voice (`es`, `en`, or provider hint)
- `style`: your style preset label
- `prosody.speed`: 0.7..1.4
- `prosody.pitch`: -1..1
- `prosody.energy`: 0..1
- `prosody.pause_ms`: 0..500
- `generation.temperature`: 0..1
- `generation.stability`: 0..1
- `generation.seed`: integer
- `generation.model`: provider model override (Groq)
- `generation.provider_voice`: provider voice override (Groq)
- `audio.format`: `wav` or `mp3` (provider-dependent)
- `constraints.max_chars`: max chars sent to TTS

Provider mapping:

- `TTS_PROVIDER=local` (Piper):
  - optional `instance_id`: choose local model instance by id
  - `speed` -> `length_scale`
  - `energy` and `pitch` -> `noise_scale` and `noise_w`
  - `pause_ms` -> `sentence_silence`
- `TTS_PROVIDER=f5` (F5 node):
  - orchestrator sends the same standardized payload
  - node maps to `ref_audio/ref_text/speed`
- `TTS_PROVIDER=groq`:
  - orchestrator also delegates to node (`tts` service)
  - node calls Groq API internally (`generation.model`, `generation.provider_voice`, `audio.format`)

Local TTS helper endpoints:

- `GET /health` -> model and service status
- `GET /instances` -> list available local voice instances

Standardized TTS node API (used by orchestrator for `local` and `f5`):
Standardized TTS node API (used by orchestrator for `local`, `f5`, and delegated `groq`):

- `GET /health`
- `GET /instances`
- `GET /config` (full persistent config + effective instances)
- `PUT /config` (update persistent config)
- `POST /config/ephemeral` (set temporary instance with TTL seconds)
- `DELETE /config/ephemeral` (clear temporary instance)
- `POST /speak` with common payload:

```json
{
  "text": "output text",
  "lang": "es-AR",
  "voice": "es",
  "instance_id": "es_ES-sharvard-medium",
  "style": "rioplatense_friendly",
  "prosody": { "speed": 0.97, "pitch": -0.05, "energy": 0.78, "pause_ms": 130 },
  "generation": { "temperature": 0.6, "stability": 0.78, "seed": 42 },
  "audio": { "format": "wav", "sample_rate": 22050 },
  "constraints": { "max_chars": 340 }
}
```

Runtime tip:

- `TTS_PROVIDER=local` -> node uses local Piper
- `TTS_PROVIDER=f5` -> orchestrator targets `tts-f5` node
- `TTS_PROVIDER=groq` -> orchestrator still targets node; node performs Groq API call
- `TTS_PROVIDER_REGISTRY=local,groq,f5` defines provider order for auto-selection.
- The orchestrator picks the first `enabled + healthy` provider from the registry.
- `TTS_PROVIDER=f5`:
  - `prosody.speed` -> `speed`
  - Voice identity comes from configured `F5_REF_AUDIO_*` + `F5_REF_TEXT_*`

## Node config pattern (persistente + efimero)

Both `tts` and `tts-f5` now use JSON config files mounted from `voice-agent/config/`:

- `voice-agent/config/tts-node.json`
- `voice-agent/config/tts-f5-node.json`

Shared pattern:

- `instances[]`: named persistent presets (model/provider/ref settings)
- `default_instance_by_lang`: default preset per language
- `ephemeral`: one temporary preset for experiments (`active`, `expires_at`, `instance`)

This lets you keep stable production presets and still test temporary voice/model setups without overwriting baseline config.

Env vars:

- `TTS_CONFIG_PATH=/app/config/tts-node.json`
- `F5_CONFIG_PATH=/app/config/tts-f5-node.json`

Docker compose mounts `./config:/app/config` for both TTS containers.

## UI admin (debug panel)

The voice HUD panel includes a simple node admin block:

- provider health and enable/disable toggles
- load node config
- edit/save persistent JSON config
- set/clear ephemeral instance with TTL

WebSocket control messages:

- `node_config_get` -> server replies `node_config_status`
- `node_config_set` -> server replies `node_config_status` + refreshed `debug_status`
- `node_ephemeral_set` -> server replies `node_config_status` + refreshed `debug_status`
- `node_ephemeral_clear` -> server replies `node_config_status` + refreshed `debug_status`

Orchestrator logs include per request:

- `audio_received`
- `ffmpeg_done`
- `stt_done`
- `llm_done`
- `tts_done`
- `ws_sent`

## Security notes

- `LLM_API_KEY` stays server-side in orchestrator env.
- Browser never receives API keys.
- You can restrict websocket origins with `ALLOWED_ORIGINS`.

## Test plan

Use these phrases from the frontend voice panel:

- Spanish: `Hola, contame quién sos y qué proyectos destacás.`
- English: `Tell me about your main projects and tech stack.`

Measure latency from logs:

- `audio_end -> ffmpeg_done -> stt_done -> llm_done -> tts_done -> first_play`

`first_play` is observable in browser when audio playback starts.
