# Voice Agent MVP (local STT + TTS, external LLM)

This folder adds a local voice-agent stack for the portfolio:

- Frontend (React + Vite): push-to-talk panel over WebSocket
- Orchestrator (Node): audio conversion + STT + LLM + TTS pipeline
- STT (FastAPI + faster-whisper): local transcription (`/transcribe`)
- TTS (FastAPI + Piper CLI): local speech synthesis (`/speak`)

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

### GPU profile (recommended on RTX 3070)

```bash
cd voice-agent
docker compose --profile gpu up --build
```

### CPU fallback profile

```bash
cd voice-agent
docker compose --profile cpu up --build
```

In another terminal (repo root):

```bash
pnpm dev
```

## Piper voices

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
3. STT `POST /transcribe` (multipart file).
4. Build persona prompt + optional CMS grounding snippets.
5. Call external LLM API (OpenAI-compatible chat-completions adapter).
6. Post-process assistant text for short TTS-friendly response.
7. TTS `POST /speak` and return WAV via WebSocket.

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
