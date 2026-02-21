# Voice Agent MVP (Groq STT + Groq LLM + Groq/API or local Piper TTS)

This folder adds a local voice-agent stack for the portfolio:

- Frontend (React + Vite): push-to-talk panel over WebSocket
- Orchestrator (Node): audio conversion + STT + LLM + TTS pipeline
- STT:
  - Groq Whisper API (`STT_PROVIDER=groq`) recommended for trial quota tests
  - Optional local fallback (FastAPI + faster-whisper) with Docker profiles
- TTS:
  - Groq API (`TTS_PROVIDER=groq`) for quick POC
  - Local Piper fallback (`TTS_PROVIDER=local`)

## Services and ports

- Orchestrator WS: `ws://localhost:8787`
- STT HTTP: `http://localhost:9001`
- TTS HTTP: `http://localhost:9002`

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

In another terminal (repo root):

```bash
pnpm dev
```

## Piper voices (used when `TTS_PROVIDER=local` or fallback)

Put Piper models in `voice-agent/tts/voices` and match paths in `.env`:

- `PIPER_VOICE_ES`
- `PIPER_VOICE_EN`

Example expected layout:

- `voice-agent/tts/voices/es/es_AR-voice.onnx`
- `voice-agent/tts/voices/en/en_US-voice.onnx`

You can use your preferred Piper voices; this repo does not bundle model weights.

## WebSocket contract

### Client -> server (two-frame audio upload)

1. Text frame metadata:

```json
{ "type": "audio_meta", "requestId": "<uuid>", "lang": "es-AR", "mime": "audio/webm;codecs=opus" }
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
5. Build persona prompt + optional CMS grounding snippets.
6. Call external LLM API (OpenAI-compatible chat-completions adapter).
7. Post-process assistant text for short TTS-friendly response.
8. Return WAV via WebSocket.

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
