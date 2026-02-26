import json
import os
import subprocess
import tempfile
import urllib.error
import urllib.request
from datetime import datetime, timedelta, timezone
from pathlib import Path
from typing import Literal

from fastapi import FastAPI, HTTPException
from fastapi.responses import Response
from pydantic import BaseModel, Field

app = FastAPI(title="minarai-tts", version="0.2.0")

PIPER_BIN = os.getenv("PIPER_BIN", "/opt/piper/piper")
PIPER_VOICE_ES = os.getenv("PIPER_VOICE_ES", "/app/voices/es/es_ES-sharvard-medium.onnx")
PIPER_VOICE_EN = os.getenv("PIPER_VOICE_EN", "/app/voices/en/en_US-joe-medium.onnx")
TTS_NODE_PROVIDER = os.getenv("TTS_NODE_PROVIDER", "local").strip().lower()
GROQ_API_KEY = (os.getenv("GROQ_API_KEY", "") or os.getenv("LLM_API_KEY", "")).strip()
GROQ_TTS_URL = os.getenv("GROQ_TTS_URL", "https://api.groq.com/openai/v1/audio/speech")
GROQ_TTS_MODEL = os.getenv("GROQ_TTS_MODEL", "playai-tts")
GROQ_TTS_VOICE = os.getenv("GROQ_TTS_VOICE", "Fritz-PlayAI")
GROQ_TTS_INPUT_MAX_CHARS = int(os.getenv("GROQ_TTS_INPUT_MAX_CHARS", "180"))
TTS_CONFIG_PATH = Path(os.getenv("TTS_CONFIG_PATH", "/app/config/tts-node.json"))


class TtsNodeInstance(BaseModel):
    id: str = Field(min_length=1, max_length=120)
    lang: Literal["es", "en"] = "es"
    provider: Literal["local", "groq"] = "local"
    enabled: bool = True
    model_path: str | None = None
    model: str | None = None
    provider_voice: str | None = None


class ConfigUpdateRequest(BaseModel):
    provider: Literal["local", "groq"] | None = None
    default_instance_by_lang: dict[str, str] | None = None
    instances: list[TtsNodeInstance] | None = None


class EphemeralSetRequest(BaseModel):
    instance: TtsNodeInstance
    ttl_seconds: int = Field(default=900, ge=30, le=86_400)


class SpeakRequest(BaseModel):
    text: str = Field(min_length=1, max_length=1200)
    voice: Literal["es", "en"] = "es"
    lang: str | None = None
    instance_id: str | None = None
    style: str | None = None
    prosody: dict | None = None
    generation: dict | None = None
    audio: dict | None = None
    constraints: dict | None = None
    provider: str | None = None
    length_scale: float | None = Field(default=None, ge=0.5, le=2.0)
    noise_scale: float | None = Field(default=None, ge=0.0, le=2.0)
    noise_w: float | None = Field(default=None, ge=0.0, le=2.0)
    sentence_silence: float | None = Field(default=None, ge=0.0, le=2.0)


_cached_config: dict | None = None


def _utc_now() -> datetime:
    return datetime.now(timezone.utc)


def _read_json(path: Path) -> dict | None:
    if not path.exists():
        return None
    try:
        return json.loads(path.read_text(encoding="utf-8"))
    except Exception:
        return None


def _write_json(path: Path, payload: dict) -> None:
    path.parent.mkdir(parents=True, exist_ok=True)
    path.write_text(json.dumps(payload, ensure_ascii=True, indent=2), encoding="utf-8")


def _fallback_voice_path(voice: str):
    directory = Path("/app/voices/en" if voice == "en" else "/app/voices/es")
    if not directory.exists():
        return None
    candidates = sorted(
        p for p in directory.glob("*.onnx") if p.is_file() and p.name.endswith(".onnx")
    )
    return str(candidates[0]) if candidates else None


def _voice_path(voice: str) -> str:
    configured = PIPER_VOICE_EN if voice == "en" else PIPER_VOICE_ES
    if Path(configured).exists():
        return configured
    fallback = _fallback_voice_path(voice)
    if fallback:
        return fallback
    return configured


def _instance_id_for(path: str) -> str:
    return Path(path).stem


def _default_config() -> dict:
    configured_es = _voice_path("es")
    configured_en = _voice_path("en")
    return {
        "version": 1,
        "provider": TTS_NODE_PROVIDER if TTS_NODE_PROVIDER in {"local", "groq"} else "local",
        "default_instance_by_lang": {
            "es": _instance_id_for(configured_es),
            "en": _instance_id_for(configured_en),
        },
        "instances": [
            {
                "id": _instance_id_for(configured_es),
                "lang": "es",
                "provider": "local",
                "enabled": True,
                "model_path": configured_es,
            },
            {
                "id": _instance_id_for(configured_en),
                "lang": "en",
                "provider": "local",
                "enabled": True,
                "model_path": configured_en,
            },
            {
                "id": "groq-default",
                "lang": "en",
                "provider": "groq",
                "enabled": True,
                "model": GROQ_TTS_MODEL,
                "provider_voice": GROQ_TTS_VOICE,
            },
        ],
        "ephemeral": {
            "active": False,
            "expires_at": None,
            "instance": None,
        },
    }


def _normalize_config(raw: dict) -> dict:
    defaults = _default_config()
    provider = str(raw.get("provider") or defaults["provider"]).lower()
    if provider not in {"local", "groq"}:
        provider = defaults["provider"]

    default_map_raw = raw.get("default_instance_by_lang") or defaults["default_instance_by_lang"]
    default_map = {
        "es": str(default_map_raw.get("es") or defaults["default_instance_by_lang"]["es"]),
        "en": str(default_map_raw.get("en") or defaults["default_instance_by_lang"]["en"]),
    }

    normalized_instances: list[dict] = []
    for item in raw.get("instances") or defaults["instances"]:
        try:
            parsed = TtsNodeInstance(**item)
            normalized_instances.append(parsed.model_dump())
        except Exception:
            continue

    ephemeral_raw = raw.get("ephemeral") or {}
    ephemeral_instance = None
    if isinstance(ephemeral_raw.get("instance"), dict):
        try:
            ephemeral_instance = TtsNodeInstance(**ephemeral_raw["instance"]).model_dump()
        except Exception:
            ephemeral_instance = None

    normalized = {
        "version": 1,
        "provider": provider,
        "default_instance_by_lang": default_map,
        "instances": normalized_instances,
        "ephemeral": {
            "active": bool(ephemeral_raw.get("active")) and ephemeral_instance is not None,
            "expires_at": ephemeral_raw.get("expires_at"),
            "instance": ephemeral_instance,
        },
    }
    return normalized


def _load_config() -> dict:
    global _cached_config
    if _cached_config is not None:
        return _cached_config
    raw = _read_json(TTS_CONFIG_PATH)
    if raw is None:
        raw = _default_config()
        _write_json(TTS_CONFIG_PATH, raw)
    normalized = _normalize_config(raw)
    _cached_config = normalized
    if normalized != raw:
        _write_json(TTS_CONFIG_PATH, normalized)
    return normalized


def _save_config(config: dict) -> dict:
    global _cached_config
    normalized = _normalize_config(config)
    _write_json(TTS_CONFIG_PATH, normalized)
    _cached_config = normalized
    return normalized


def _ephemeral_instance(config: dict) -> dict | None:
    ephemeral = config.get("ephemeral") or {}
    if not ephemeral.get("active"):
        return None

    expires_at = ephemeral.get("expires_at")
    if isinstance(expires_at, str) and expires_at.strip():
        try:
            expiry = datetime.fromisoformat(expires_at.replace("Z", "+00:00"))
            if _utc_now() >= expiry:
                config["ephemeral"] = {"active": False, "expires_at": None, "instance": None}
                _save_config(config)
                return None
        except ValueError:
            config["ephemeral"] = {"active": False, "expires_at": None, "instance": None}
            _save_config(config)
            return None

    instance = ephemeral.get("instance")
    if isinstance(instance, dict):
        return instance
    return None


def _discovered_local_instances() -> list[dict]:
    discovered = []
    for lang in ["es", "en"]:
        directory = Path("/app/voices") / lang
        if not directory.exists():
            continue
        for file in sorted(directory.glob("*.onnx")):
            discovered.append(
                {
                    "id": _instance_id_for(str(file)),
                    "lang": lang,
                    "provider": "local",
                    "enabled": True,
                    "model_path": str(file),
                    "source": "discovered",
                }
            )
    return discovered


def _all_instances(config: dict) -> list[dict]:
    items = []
    seen = set()

    for instance in config.get("instances", []):
        if not instance.get("enabled", True):
            continue
        key = str(instance.get("id"))
        if key in seen:
            continue
        seen.add(key)
        items.append({**instance, "source": "configured"})

    for instance in _discovered_local_instances():
        key = str(instance.get("id"))
        if key in seen:
            continue
        seen.add(key)
        items.append(instance)

    ephemeral = _ephemeral_instance(config)
    if ephemeral:
        items.append({**ephemeral, "id": "ephemeral", "source": "ephemeral"})

    return items


def _resolve_instance(config: dict, voice: str, instance_id: str | None) -> dict:
    instances = _all_instances(config)
    if instance_id:
        for instance in instances:
            if instance.get("id") == instance_id:
                return instance
        raise HTTPException(status_code=400, detail=f"Unknown instance_id: {instance_id}")

    default_map = config.get("default_instance_by_lang") or {}
    default_id = str(default_map.get("en" if voice == "en" else "es") or "")
    if default_id:
        for instance in instances:
            if instance.get("id") == default_id:
                return instance

    for instance in instances:
        if instance.get("lang") == voice:
            return instance

    return {
        "id": "fallback",
        "lang": voice,
        "provider": "local",
        "enabled": True,
        "model_path": _voice_path(voice),
        "source": "fallback",
    }


def _resolve_numeric(req: SpeakRequest, name: str, default: float) -> float:
    direct_value = getattr(req, name, None)
    if direct_value is not None:
        return float(direct_value)
    if isinstance(req.prosody, dict):
        legacy_key = "pause_ms" if name == "sentence_silence" else name
        raw = req.prosody.get(legacy_key)
        if raw is not None:
            try:
                parsed = float(raw)
                if name == "sentence_silence":
                    return max(0.0, min(2.0, parsed if parsed <= 2 else parsed / 1000))
                if name == "length_scale":
                    return max(0.5, min(2.0, 1 / max(0.5, min(2.0, parsed))))
                return max(0.0, min(2.0, parsed))
            except (TypeError, ValueError):
                pass
    return default


def _resolve_provider(req: SpeakRequest, instance: dict, config: dict) -> str:
    provider = (req.provider or instance.get("provider") or config.get("provider") or "local").strip().lower()
    if provider in {"local", "groq"}:
        return provider
    return "local"


def _groq_speak(req: SpeakRequest, instance: dict) -> bytes:
    if not GROQ_API_KEY:
        raise HTTPException(status_code=500, detail="Groq API key missing in TTS node")

    generation = req.generation if isinstance(req.generation, dict) else {}
    audio = req.audio if isinstance(req.audio, dict) else {}
    constraints = req.constraints if isinstance(req.constraints, dict) else {}

    model = str(generation.get("model") or instance.get("model") or GROQ_TTS_MODEL)
    provider_voice = str(
        generation.get("provider_voice") or instance.get("provider_voice") or GROQ_TTS_VOICE
    )
    response_format = str(audio.get("format") or "wav")
    max_chars = int(constraints.get("max_chars") or GROQ_TTS_INPUT_MAX_CHARS)
    input_text = req.text[: max(1, min(max_chars, 4000))].strip()

    payload = {
        "model": model,
        "voice": provider_voice,
        "input": input_text,
        "response_format": response_format,
    }
    data = json.dumps(payload).encode("utf-8")
    request = urllib.request.Request(
        GROQ_TTS_URL,
        data=data,
        method="POST",
        headers={
            "Content-Type": "application/json",
            "Authorization": f"Bearer {GROQ_API_KEY}",
        },
    )
    try:
        with urllib.request.urlopen(request, timeout=20) as response:
            return response.read()
    except urllib.error.HTTPError as exc:
        body = exc.read().decode("utf-8", errors="ignore")
        raise HTTPException(
            status_code=500, detail=f"Groq TTS failed: HTTP {exc.code}: {body[:400]}"
        ) from exc
    except Exception as exc:  # noqa: BLE001
        raise HTTPException(status_code=500, detail=f"Groq TTS failed: {exc}") from exc


@app.get("/health")
def health() -> dict[str, object]:
    config = _load_config()
    selected_es = _voice_path("es")
    selected_en = _voice_path("en")
    return {
        "ok": True,
        "piper_bin": PIPER_BIN,
        "voice_es_exists": Path(PIPER_VOICE_ES).exists(),
        "voice_en_exists": Path(PIPER_VOICE_EN).exists(),
        "voice_es_selected": selected_es,
        "voice_en_selected": selected_en,
        "voice_es_selected_exists": Path(selected_es).exists(),
        "voice_en_selected_exists": Path(selected_en).exists(),
        "tts_node_provider": config.get("provider"),
        "groq_key_present": bool(GROQ_API_KEY),
        "groq_tts_model": GROQ_TTS_MODEL,
        "groq_tts_voice": GROQ_TTS_VOICE,
        "config_path": str(TTS_CONFIG_PATH),
        "ephemeral_active": bool(_ephemeral_instance(config)),
    }


@app.get("/instances")
def instances() -> dict[str, object]:
    config = _load_config()
    return {
        "ok": True,
        "provider": config.get("provider"),
        "instances": _all_instances(config),
        "default_instance_by_lang": config.get("default_instance_by_lang"),
    }


@app.get("/config")
def get_config() -> dict[str, object]:
    config = _load_config()
    return {
        "ok": True,
        "config": config,
        "instances": _all_instances(config),
    }


@app.put("/config")
def update_config(req: ConfigUpdateRequest) -> dict[str, object]:
    config = _load_config()

    if req.provider is not None:
        config["provider"] = req.provider
    if req.default_instance_by_lang is not None:
        config["default_instance_by_lang"] = {
            "es": str(req.default_instance_by_lang.get("es") or config.get("default_instance_by_lang", {}).get("es") or ""),
            "en": str(req.default_instance_by_lang.get("en") or config.get("default_instance_by_lang", {}).get("en") or ""),
        }
    if req.instances is not None:
        config["instances"] = [instance.model_dump() for instance in req.instances]

    saved = _save_config(config)
    return {"ok": True, "config": saved, "instances": _all_instances(saved)}


@app.post("/config/ephemeral")
def set_ephemeral(req: EphemeralSetRequest) -> dict[str, object]:
    config = _load_config()
    expires_at = (_utc_now() + timedelta(seconds=req.ttl_seconds)).isoformat()
    config["ephemeral"] = {
        "active": True,
        "expires_at": expires_at,
        "instance": req.instance.model_dump(),
    }
    saved = _save_config(config)
    return {
        "ok": True,
        "ephemeral": saved.get("ephemeral"),
        "instances": _all_instances(saved),
    }


@app.delete("/config/ephemeral")
def clear_ephemeral() -> dict[str, object]:
    config = _load_config()
    config["ephemeral"] = {"active": False, "expires_at": None, "instance": None}
    saved = _save_config(config)
    return {"ok": True, "ephemeral": saved.get("ephemeral"), "instances": _all_instances(saved)}


@app.post("/speak")
def speak(req: SpeakRequest):
    config = _load_config()
    instance = _resolve_instance(config, req.voice, req.instance_id)
    provider = _resolve_provider(req, instance, config)

    if provider == "groq":
        wav_data = _groq_speak(req, instance)
        return Response(content=wav_data, media_type="audio/wav")

    voice_model = str(instance.get("model_path") or _voice_path(req.voice))
    if not Path(PIPER_BIN).exists():
        raise HTTPException(status_code=500, detail=f"Piper binary not found: {PIPER_BIN}")
    if not Path(voice_model).exists():
        raise HTTPException(status_code=500, detail=f"Voice model missing: {voice_model}")

    with tempfile.NamedTemporaryFile(delete=False, suffix=".wav") as tmp:
        output_path = tmp.name

    cmd = [
        PIPER_BIN,
        "--model",
        voice_model,
        "--output_file",
        output_path,
    ]
    cmd += ["--length_scale", str(_resolve_numeric(req, "length_scale", 1.0))]
    cmd += ["--noise_scale", str(_resolve_numeric(req, "noise_scale", 0.667))]
    cmd += ["--noise_w", str(_resolve_numeric(req, "noise_w", 0.8))]
    cmd += ["--sentence_silence", str(_resolve_numeric(req, "sentence_silence", 0.12))]

    try:
        run = subprocess.run(
            cmd,
            input=req.text.encode("utf-8"),
            stdout=subprocess.PIPE,
            stderr=subprocess.PIPE,
            check=False,
        )

        if run.returncode != 0:
            stderr = run.stderr.decode("utf-8", errors="ignore")
            raise HTTPException(status_code=500, detail=f"Piper synthesis failed: {stderr[:400]}")

        wav_data = Path(output_path).read_bytes()
        return Response(content=wav_data, media_type="audio/wav")
    finally:
        try:
            os.remove(output_path)
        except OSError:
            pass
