"""
Morph Backend — detection routes.

POST /api/detection/analyze

Handles browser-recorded WebM/OGG audio by converting to WAV via FFmpeg
before passing to ai_engine.
"""

from __future__ import annotations

import asyncio
import logging
import shutil
import subprocess
import tempfile
from pathlib import Path

import ffmpeg
from fastapi import APIRouter, File, HTTPException, UploadFile

from app.config import PROJECT_ROOT, SUPPORTED_AUDIO_EXTENSIONS

log = logging.getLogger("morph-backend.detection")

router = APIRouter(prefix="/api/detection", tags=["detection"])

# ---------------------------------------------------------------------------
# FFmpeg availability check
# ---------------------------------------------------------------------------

_FFMPEG_AVAILABLE = shutil.which("ffmpeg") is not None

if not _FFMPEG_AVAILABLE:
    log.warning(
        "ffmpeg not found on PATH — browser audio (WebM/OGG/MP3) "
        "conversion will fail. Install ffmpeg to support all formats."
    )


def _check_ffmpeg() -> None:
    """Raise early if ffmpeg is missing and a conversion would be needed."""
    if not _FFMPEG_AVAILABLE:
        raise HTTPException(
            status_code=500,
            detail=(
                "FFmpeg is not installed on the server. "
                "Browser-recorded audio (WebM/OGG) cannot be converted. "
                "Install ffmpeg and restart the server."
            ),
        )


# ---------------------------------------------------------------------------
# Conversion helpers
# ---------------------------------------------------------------------------

# Formats that need conversion before ai_engine can read them
_NEEDS_CONVERSION = {".webm", ".ogg", ".mp3", ".m4a", ".aac", ".opus"}


def _convert_to_wav(input_path: str, suffix: str) -> str:
    """
    Convert an audio file to mono PCM WAV at its **native sample rate** using FFmpeg.

    IMPORTANT: FFmpeg must NOT resample here — librosa handles resampling
    to 16 kHz internally during feature extraction. FFmpeg resampling
    changes the spectral特征特征 in a way that diverges from the
    training data distribution and causes model misclassification.

    Returns the path to the converted WAV file.
    """
    _check_ffmpeg()

    out = tempfile.NamedTemporaryFile(suffix=".wav", delete=False, dir=str(PROJECT_ROOT / "data" / "interim"))
    out_path = out.name
    out.close()

    try:
        (
            ffmpeg.input(input_path)
            .output(
                out_path,
                acodec="pcm_s16le",
                ac="1",
            )
            .overwrite_output()
            .run(capture_stdout=True, capture_stderr=True, quiet=True)
        )
    except ffmpeg.Error as exc:
        # Clean up partial output
        Path(out_path).unlink(missing_ok=True)
        stderr = exc.stderr.decode(errors="replace") if exc.stderr else "unknown error"
        raise RuntimeError(f"FFmpeg conversion failed: {stderr}") from exc

    return out_path


# ---------------------------------------------------------------------------
# Detection (synchronous, runs in thread executor)
# ---------------------------------------------------------------------------


def _run_detection_sync(audio_path: str) -> dict:
    """Run detection synchronously — called in a thread executor."""
    from ai_engine.inference.pipeline import detect

    result = detect(audio_path, model_version="v2")
    return {
        "verdict": result.label_str,
        "risk_score": round(result.fake_probability * 100, 2),
        "real_probability": round(result.real_probability, 6),
        "fake_probability": round(result.fake_probability, 6),
        "model_version": result.model_version,
        "duration": round(result.duration, 2),
        "feature_count": 132,
    }


# ---------------------------------------------------------------------------
# Route
# ---------------------------------------------------------------------------


@router.post("/analyze")
async def analyze_audio(
    file: UploadFile = File(..., description="Audio file (WAV/FLAC/WebM/OGG/MP3)."),
):
    """
    Accept an uploaded audio file, run Morph V2 detection, and return results.

    - WAV/FLAC: passed directly to ai_engine.
    - WebM/OGG/MP3/etc: converted to 16 kHz mono WAV via FFmpeg first.
    - All temporary files are cleaned up after inference.
    """
    # ── Validate file ──────────────────────────────────────
    if not file.filename:
        raise HTTPException(status_code=400, detail="No filename provided.")

    ext = Path(file.filename).suffix.lower()
    if ext not in SUPPORTED_AUDIO_EXTENSIONS:
        raise HTTPException(
            status_code=400,
            detail=f"Unsupported format '{ext}'. Supported: {', '.join(sorted(SUPPORTED_AUDIO_EXTENSIONS))}",
        )

    # ── Read file bytes ────────────────────────────────────
    contents = await file.read()
    if not contents:
        raise HTTPException(status_code=400, detail="Empty file.")

    max_bytes = 50 * 1024 * 1024  # 50 MB
    if len(contents) > max_bytes:
        raise HTTPException(status_code=413, detail="File too large (max 50 MB).")

    # ── Save upload to temp file ───────────────────────────
    upload_tmp = None
    converted_tmp_path = None

    try:
        upload_tmp = tempfile.NamedTemporaryFile(
            suffix=ext, delete=False, dir=str(PROJECT_ROOT / "data" / "interim")
        )
        upload_tmp.write(contents)
        upload_tmp.flush()
        upload_tmp.close()

        # ── Convert if needed ──────────────────────────────
        if ext in _NEEDS_CONVERSION:
            log.info("Converting %s → WAV via FFmpeg", ext)
            try:
                converted_tmp_path = _convert_to_wav(upload_tmp.name, ext)
            except RuntimeError as exc:
                log.exception("FFmpeg conversion failed")
                raise HTTPException(
                    status_code=422,
                    detail=f"Could not decode audio: {exc}",
                ) from exc
            detect_path = converted_tmp_path
        else:
            detect_path = upload_tmp.name

        # ── Run inference in thread executor ───────────────
        result = await asyncio.get_event_loop().run_in_executor(
            None, _run_detection_sync, detect_path
        )

        return result

    except HTTPException:
        raise
    except Exception as exc:
        log.exception("Detection failed")
        raise HTTPException(
            status_code=500, detail=f"Detection failed: {exc}"
        ) from exc
    finally:
        # ── Clean up all temp files ────────────────────────
        for p in (upload_tmp, converted_tmp_path):
            if p is not None:
                try:
                    Path(p.name if hasattr(p, "name") else p).unlink(missing_ok=True)
                except OSError:
                    pass
