import os
import subprocess
import tempfile
from pathlib import Path
from typing import Literal

from fastapi import FastAPI, HTTPException
from fastapi.responses import Response
from pydantic import BaseModel, Field

app = FastAPI(title="minarai-tts", version="0.1.0")

PIPER_BIN = os.getenv("PIPER_BIN", "/opt/piper/piper")
PIPER_VOICE_ES = os.getenv("PIPER_VOICE_ES", "/app/voices/es/model.onnx")
PIPER_VOICE_EN = os.getenv("PIPER_VOICE_EN", "/app/voices/en/model.onnx")


class SpeakRequest(BaseModel):
    text: str = Field(min_length=1, max_length=1200)
    voice: Literal["es", "en"] = "es"


def _voice_path(voice: str) -> str:
    return PIPER_VOICE_EN if voice == "en" else PIPER_VOICE_ES


@app.get("/health")
def health() -> dict[str, object]:
    return {
        "ok": True,
        "piper_bin": PIPER_BIN,
        "voice_es_exists": Path(PIPER_VOICE_ES).exists(),
        "voice_en_exists": Path(PIPER_VOICE_EN).exists(),
    }


@app.post("/speak")
def speak(req: SpeakRequest):
    voice_model = _voice_path(req.voice)
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
