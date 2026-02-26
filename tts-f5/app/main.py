import json
import os
import tempfile
from datetime import datetime, timedelta, timezone
from pathlib import Path
from typing import Literal

from fastapi import FastAPI, HTTPException
from fastapi.responses import Response
from pydantic import BaseModel, Field

app = FastAPI(title="minarai-tts-f5", version="0.2.0")

F5_MODEL_ID = os.getenv("F5_MODEL_ID", "jpgallegoar/F5-Spanish")
F5_MODEL_LOCAL_PATH = os.getenv("F5_MODEL_LOCAL_PATH", "").strip()
F5_DEVICE = os.getenv("F5_DEVICE", "cuda")
F5_REF_AUDIO_ES = os.getenv("F5_REF_AUDIO_ES", "/app/voices/es/reference.wav")
F5_REF_TEXT_ES = os.getenv("F5_REF_TEXT_ES", "").strip()
F5_REF_AUDIO_EN = os.getenv("F5_REF_AUDIO_EN", "").strip()
F5_REF_TEXT_EN = os.getenv("F5_REF_TEXT_EN", "").strip()
F5_DEFAULT_SPEED = float(os.getenv("F5_DEFAULT_SPEED", "1.0"))
F5_CONFIG_PATH = Path(os.getenv("F5_CONFIG_PATH", "/app/config/tts-f5-node.json"))

_tts_cache: dict[str, object] = {}
_tts_error_cache: dict[str, str] = {}


class F5Instance(BaseModel):
    id: str = Field(min_length=1, max_length=120)
    lang: Literal["es", "en"] = "es"
    enabled: bool = True
    model_id: str | None = None
    model_local_path: str | None = None
    ref_audio: str | None = None
    ref_text: str | None = None


class ConfigUpdateRequest(BaseModel):
    default_instance_by_lang: dict[str, str] | None = None
    instances: list[F5Instance] | None = None


class EphemeralSetRequest(BaseModel):
    instance: F5Instance
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
    speed: float | None = Field(default=None, ge=0.6, le=1.4)
    ref_audio: str | None = None
    ref_text: str | None = None


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


def _default_config() -> dict:
    default_es = {
        "id": "f5-es-default",
        "lang": "es",
        "enabled": True,
        "model_id": F5_MODEL_ID,
        "model_local_path": F5_MODEL_LOCAL_PATH or None,
        "ref_audio": F5_REF_AUDIO_ES,
        "ref_text": F5_REF_TEXT_ES,
    }
    instances = [default_es]
    if F5_REF_AUDIO_EN:
        instances.append(
            {
                "id": "f5-en-default",
                "lang": "en",
                "enabled": True,
                "model_id": F5_MODEL_ID,
                "model_local_path": F5_MODEL_LOCAL_PATH or None,
                "ref_audio": F5_REF_AUDIO_EN,
                "ref_text": F5_REF_TEXT_EN,
            }
        )

    return {
        "version": 1,
        "default_instance_by_lang": {
            "es": "f5-es-default",
            "en": "f5-en-default" if F5_REF_AUDIO_EN else "f5-es-default",
        },
        "instances": instances,
        "ephemeral": {
            "active": False,
            "expires_at": None,
            "instance": None,
        },
    }


def _normalize_config(raw: dict) -> dict:
    defaults = _default_config()
    default_map_raw = raw.get("default_instance_by_lang") or defaults["default_instance_by_lang"]
    default_map = {
        "es": str(default_map_raw.get("es") or defaults["default_instance_by_lang"]["es"]),
        "en": str(default_map_raw.get("en") or defaults["default_instance_by_lang"]["en"]),
    }

    normalized_instances: list[dict] = []
    for item in raw.get("instances") or defaults["instances"]:
        try:
            parsed = F5Instance(**item)
            normalized_instances.append(parsed.model_dump())
        except Exception:
            continue

    ephemeral_raw = raw.get("ephemeral") or {}
    ephemeral_instance = None
    if isinstance(ephemeral_raw.get("instance"), dict):
        try:
            ephemeral_instance = F5Instance(**ephemeral_raw["instance"]).model_dump()
        except Exception:
            ephemeral_instance = None

    return {
        "version": 1,
        "default_instance_by_lang": default_map,
        "instances": normalized_instances,
        "ephemeral": {
            "active": bool(ephemeral_raw.get("active")) and ephemeral_instance is not None,
            "expires_at": ephemeral_raw.get("expires_at"),
            "instance": ephemeral_instance,
        },
    }


def _load_config() -> dict:
    global _cached_config
    if _cached_config is not None:
        return _cached_config
    raw = _read_json(F5_CONFIG_PATH)
    if raw is None:
        raw = _default_config()
        _write_json(F5_CONFIG_PATH, raw)
    normalized = _normalize_config(raw)
    _cached_config = normalized
    if normalized != raw:
        _write_json(F5_CONFIG_PATH, normalized)
    return normalized


def _save_config(config: dict) -> dict:
    global _cached_config
    normalized = _normalize_config(config)
    _write_json(F5_CONFIG_PATH, normalized)
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


def _all_instances(config: dict) -> list[dict]:
    out = []
    for item in config.get("instances", []):
        if item.get("enabled", True):
            out.append({**item, "source": "configured"})

    eph = _ephemeral_instance(config)
    if eph:
        out.append({**eph, "id": "ephemeral", "source": "ephemeral"})
    return out


def _resolve_instance(config: dict, voice: str, instance_id: str | None) -> dict:
    instances = _all_instances(config)
    if instance_id:
        for item in instances:
            if item.get("id") == instance_id:
                return item
        raise HTTPException(status_code=400, detail=f"Unknown instance_id: {instance_id}")

    defaults = config.get("default_instance_by_lang") or {}
    wanted = str(defaults.get("en" if voice == "en" else "es") or "")
    if wanted:
        for item in instances:
            if item.get("id") == wanted:
                return item

    for item in instances:
        if item.get("lang") == voice:
            return item

    raise HTTPException(status_code=500, detail="No enabled F5 instances configured")


def _resolve_speed(req: SpeakRequest) -> float:
    if req.speed is not None:
        return req.speed
    if isinstance(req.prosody, dict):
        raw = req.prosody.get("speed")
        if raw is not None:
            try:
                parsed = float(raw)
                return max(0.6, min(1.4, parsed))
            except (TypeError, ValueError):
                pass
    return F5_DEFAULT_SPEED


def _download_model_snapshot(model_id: str) -> str:
    from huggingface_hub import snapshot_download

    return snapshot_download(repo_id=model_id)


def _pick_ckpt_and_vocab(model_dir: Path) -> tuple[str, str]:
    ckpt_candidates = sorted(model_dir.glob("*.safetensors")) + sorted(model_dir.glob("*.pt"))
    vocab_candidates = sorted(model_dir.glob("vocab*.txt")) + sorted(model_dir.glob("*.vocab"))
    if not ckpt_candidates:
        raise RuntimeError(
            f"No checkpoint file found in {model_dir}. Expected .safetensors or .pt from F5 model."
        )
    if not vocab_candidates:
        raise RuntimeError(
            f"No vocab file found in {model_dir}. Expected vocab*.txt (or *.vocab) from F5 model."
        )
    return str(ckpt_candidates[0]), str(vocab_candidates[0])


def _ensure_model(instance: dict):
    model_key = str(instance.get("model_local_path") or instance.get("model_id") or F5_MODEL_ID)
    if model_key in _tts_cache:
        return _tts_cache[model_key], model_key
    if model_key in _tts_error_cache:
        raise RuntimeError(_tts_error_cache[model_key])

    try:
        from f5_tts.api import F5TTS

        model_local_path = str(instance.get("model_local_path") or "").strip()
        model_id = str(instance.get("model_id") or F5_MODEL_ID).strip()
        model_path = Path(model_local_path) if model_local_path else Path(_download_model_snapshot(model_id))
        ckpt_file, vocab_file = _pick_ckpt_and_vocab(model_path)

        tts = F5TTS(
            model_type="F5-TTS",
            ckpt_file=ckpt_file,
            vocab_file=vocab_file,
            local_path=str(model_path),
            device=F5_DEVICE,
        )
        _tts_cache[model_key] = tts
        return tts, model_key
    except Exception as exc:  # noqa: BLE001
        _tts_error_cache[model_key] = str(exc)
        raise


@app.get("/health")
def health() -> dict[str, object]:
    config = _load_config()
    instances = _all_instances(config)
    model_paths = [
        str(item.get("model_local_path") or item.get("model_id") or F5_MODEL_ID) for item in instances
    ]
    return {
        "ok": True,
        "model_loaded_keys": list(_tts_cache.keys()),
        "load_error": _tts_error_cache,
        "device": F5_DEVICE,
        "config_path": str(F5_CONFIG_PATH),
        "ephemeral_active": bool(_ephemeral_instance(config)),
        "instances": instances,
        "configured_model_keys": model_paths,
    }


@app.get("/instances")
def instances() -> dict[str, object]:
    config = _load_config()
    return {
        "ok": True,
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

    if req.default_instance_by_lang is not None:
        current = config.get("default_instance_by_lang") or {}
        config["default_instance_by_lang"] = {
            "es": str(req.default_instance_by_lang.get("es") or current.get("es") or "f5-es-default"),
            "en": str(req.default_instance_by_lang.get("en") or current.get("en") or "f5-es-default"),
        }
    if req.instances is not None:
        config["instances"] = [item.model_dump() for item in req.instances]

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

    ref_audio = (req.ref_audio or instance.get("ref_audio") or "").strip()
    ref_text = (req.ref_text or instance.get("ref_text") or "").strip()

    if not ref_audio:
        raise HTTPException(
            status_code=500,
            detail=(
                f"Reference audio not configured for voice={req.voice}. "
                "Set ref_audio in instance config or pass ref_audio in request."
            ),
        )
    if not Path(ref_audio).exists():
        raise HTTPException(status_code=500, detail=f"Reference audio missing: {ref_audio}")
    if not ref_text:
        raise HTTPException(
            status_code=500,
            detail=(
                f"Reference text is empty for voice={req.voice}. "
                "Set ref_text in instance config or pass ref_text in request."
            ),
        )

    try:
        tts, _ = _ensure_model(instance)
    except Exception as exc:  # noqa: BLE001
        raise HTTPException(status_code=500, detail=f"F5 model init failed: {exc}") from exc

    speed = _resolve_speed(req)

    with tempfile.NamedTemporaryFile(delete=False, suffix=".wav") as tmp:
        output_path = tmp.name

    try:
        tts.infer(
            ref_file=ref_audio,
            ref_text=ref_text,
            gen_text=req.text,
            file_wave=output_path,
            speed=speed,
        )
        wav_data = Path(output_path).read_bytes()
        return Response(content=wav_data, media_type="audio/wav")
    except Exception as exc:  # noqa: BLE001
        raise HTTPException(status_code=500, detail=f"F5 synthesis failed: {exc}") from exc
    finally:
        try:
            os.remove(output_path)
        except OSError:
            pass
