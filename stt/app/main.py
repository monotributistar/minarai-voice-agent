import logging
import os
import tempfile
from typing import Any

from fastapi import FastAPI, File, HTTPException, UploadFile
from faster_whisper import WhisperModel

app = FastAPI(title="minarai-stt", version="0.1.0")

logger = logging.getLogger("stt")
logging.basicConfig(level=logging.INFO)

WHISPER_MODEL = os.getenv("WHISPER_MODEL", "medium")
WHISPER_COMPUTE_TYPE = os.getenv("WHISPER_COMPUTE_TYPE", "float16")
WHISPER_DEVICE = os.getenv("WHISPER_DEVICE", "auto")

_model: WhisperModel | None = None


def _init_model() -> WhisperModel:
    global _model

    if _model is not None:
        return _model

    preferred_device = "cuda" if WHISPER_DEVICE in ("auto", "cuda") else "cpu"

    try:
        logger.info(
            "Loading faster-whisper model=%s device=%s compute_type=%s",
            WHISPER_MODEL,
            preferred_device,
            WHISPER_COMPUTE_TYPE,
        )
        _model = WhisperModel(
            WHISPER_MODEL,
            device=preferred_device,
            compute_type=WHISPER_COMPUTE_TYPE,
        )
        return _model
    except Exception as err:
        if preferred_device == "cuda":
            logger.warning("CUDA unavailable, falling back to CPU: %s", err)
            _model = WhisperModel(
                WHISPER_MODEL,
                device="cpu",
                compute_type="int8",
            )
            return _model
        raise


@app.get("/health")
def health() -> dict[str, Any]:
    return {
        "ok": True,
        "model": WHISPER_MODEL,
        "compute_type": WHISPER_COMPUTE_TYPE,
        "device": WHISPER_DEVICE,
    }


@app.post("/transcribe")
async def transcribe(file: UploadFile = File(...)) -> dict[str, Any]:
    if not file.filename:
        raise HTTPException(status_code=400, detail="Missing filename")

    model = _init_model()

    suffix = ".wav" if file.filename.lower().endswith(".wav") else ".audio"
    with tempfile.NamedTemporaryFile(delete=False, suffix=suffix) as tmp:
        content = await file.read()
        tmp.write(content)
        tmp_path = tmp.name

    try:
        segments, info = model.transcribe(
            tmp_path,
            vad_filter=True,
            language=None,
            beam_size=4,
            condition_on_previous_text=False,
        )

        segments_list = []
        full_text = []
        for seg in segments:
            text = seg.text.strip()
            if text:
                full_text.append(text)
            segments_list.append(
                {
                    "start": float(seg.start),
                    "end": float(seg.end),
                    "text": text,
                }
            )

        return {
            "text": " ".join(full_text).strip(),
            "language": info.language,
            "segments": segments_list,
        }
    except Exception as err:
        logger.exception("Transcription failed")
        raise HTTPException(status_code=500, detail=f"Transcription failed: {err}") from err
    finally:
        try:
            os.remove(tmp_path)
        except OSError:
            pass
