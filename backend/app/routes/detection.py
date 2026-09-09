"""
Morph Backend — detection routes.

POST /api/detection/analyze
    Upload-flow detection (browser-recorded WebM/OGG → WAV via FFmpeg).

POST /api/detection/analyze-windowed
    Windowed detection with temporal aggregation.

WS  /api/detection/ws
    Live streaming detection: the client streams base64-encoded mono
    float32 PCM chunks (full precision — 16-bit quantization measurably
    flips some V2 verdicts) and receives a `detection_result` per window.
"""

from __future__ import annotations

import asyncio
import base64
import json
import logging
import shutil
import tempfile
from pathlib import Path

import ffmpeg
import numpy as np
from fastapi import (
    APIRouter,
    File,
    Form,
    HTTPException,
    Query,
    UploadFile,
    WebSocket,
    WebSocketDisconnect,
)

from app.config import PROJECT_ROOT, SUPPORTED_AUDIO_EXTENSIONS

from ai_engine.inference.pipeline import DetectionResult, detect_from_array
from ai_engine.inference.streaming import AudioWindowBuffer

log = logging.getLogger("morph-backend.detection")

router = APIRouter(prefix="/api/detection", tags=["detection"])

# ---------------------------------------------------------------------------
# Live-streaming window configuration
# ---------------------------------------------------------------------------
# 4 s windows with a 1 s hop matches the Deep-Voice training chunk length
# and yields one prediction roughly every second after the first window
# fills. Feature extraction measures ~180 ms per 4 s window on this
# hardware, so inference keeps up with real time easily.
STREAM_WINDOW_SECONDS = 4.0
STREAM_HOP_SECONDS = 1.0

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


def _run_detection_sync(audio_path: str, model_version: str = "v2_robust") -> dict:
    """Run detection synchronously — called in a thread executor."""
    if model_version == "w2v2_aasist":
        import librosa
        from ai_engine.inference.w2v2_aasist import TARGET_SR, get_w2v2_aasist

        # Load audio as 16k mono and run W2V2-AASIST (64600 pad/truncate inside wrapper)
        # Use process-level cached singleton — avoids reloading 1.2GB checkpoint per request.
        y, sr = librosa.load(audio_path, sr=TARGET_SR, mono=True)
        # duration for response
        import librosa as _lb
        duration = float(_lb.get_duration(y=y, sr=TARGET_SR))
        model = get_w2v2_aasist()
        p_real, p_fake = model.predict_proba(y, sr)
        label = 1 if p_fake >= 0.5 else 0
        return {
            "verdict": "FAKE" if label == 1 else "REAL",
            "risk_score": round(float(p_fake) * 100, 2),
            "real_probability": round(float(p_real), 6),
            "fake_probability": round(float(p_fake), 6),
            "model_version": "w2v2_aasist",
            "duration": round(duration, 2),
            "feature_count": 64600,
        }

    from ai_engine.inference.pipeline import detect

    result = detect(audio_path, model_version=model_version)
    return {
        "verdict": result.label_str,
        "risk_score": round(result.fake_probability * 100, 2),
        "real_probability": round(result.real_probability, 6),
        "fake_probability": round(result.fake_probability, 6),
        "model_version": result.model_version,
        "duration": round(result.duration, 2),
        "feature_count": 132,
    }


def _run_windowed_detection_sync(
    audio_path: str,
    window_sec: float = 4.0,
    hop_sec: float | None = None,
    aggregation_method: str = "mean",
    model_version: str = "v2_robust",
) -> dict:
    """Run windowed detection synchronously — called in thread executor."""
    if model_version == "w2v2_aasist":
        # For W2V2-AASIST, reuse the same wrapper but emulate windowed output:
        # Use the W2V2 4.04s windowing (64600) for consistency; for longer files
        # we split into 64600-sample windows and aggregate.
        import librosa
        import numpy as np
        from ai_engine.inference.w2v2_aasist import TARGET_LEN, TARGET_SR, get_w2v2_aasist
        from ai_engine.inference.windowing import get_window_boundaries
        from ai_engine.inference.aggregation import aggregate

        y, sr = librosa.load(audio_path, sr=TARGET_SR, mono=True)
        total_duration = float(librosa.get_duration(y=y, sr=TARGET_SR))
        # Use 4.0s windows for display consistency, but W2V2 internally pads to 64600
        # We'll split y into 4s windows and run W2V2 per window — reuse singleton to avoid per-window reload.
        slices = get_window_boundaries(len(y), sr=TARGET_SR, window_sec=window_sec, hop_sec=hop_sec or window_sec)
        model = get_w2v2_aasist()
        windows = []
        fake_probs = []
        for sl in slices:
            chunk = y[sl.start_sample : sl.end_sample]
            # W2V2 wrapper will pad/truncate chunk to 64600 internally
            p_real, p_fake = model.predict_proba(chunk, TARGET_SR)
            label = 1 if p_fake >= 0.5 else 0
            # duration for window
            dur = sl.duration_sec
            windows.append({
                "window_index": sl.index,
                "window": sl.index + 1,
                "start_sec": sl.start_sec,
                "end_sec": sl.end_sec,
                "duration": dur,
                "chunk_duration": dur,
                "is_partial": sl.is_partial,
                "label": label,
                "label_str": "FAKE" if label == 1 else "REAL",
                "confidence": round(float(max(p_real, p_fake)), 6),
                "real_probability": round(float(p_real), 6),
                "fake_probability": round(float(p_fake), 6),
                "real_prob": round(float(p_real), 6),
                "fake_prob": round(float(p_fake), 6),
                "model_version": "w2v2_aasist",
            })
            fake_probs.append(float(p_fake))

        # Fallback for very short file (< window)
        if not windows:
            p_real, p_fake = model.predict_proba(y, TARGET_SR)
            label = 1 if p_fake >= 0.5 else 0
            windows = [{
                "window_index": 0, "window": 1, "start_sec": 0.0, "end_sec": round(total_duration, 4),
                "duration": total_duration, "chunk_duration": total_duration, "is_partial": True,
                "label": label, "label_str": "FAKE" if label == 1 else "REAL",
                "confidence": round(float(max(p_real, p_fake)), 6),
                "real_probability": round(float(p_real), 6), "fake_probability": round(float(p_fake), 6),
                "real_prob": round(float(p_real), 6), "fake_prob": round(float(p_fake), 6),
                "model_version": "w2v2_aasist",
            }]
            fake_probs = [float(p_fake)]

        # Temporal aggregation
        agg = aggregate(fake_probs, method=aggregation_method)  # type: ignore[arg-type]
        # Full-file single prediction for comparison
        p_real_full, p_fake_full = model.predict_proba(y, TARGET_SR)
        label_full = 1 if p_fake_full >= 0.5 else 0
        return {
            "total_duration": total_duration,
            "duration": total_duration,
            "window_sec": window_sec,
            "hop_sec": hop_sec or window_sec,
            "model_version": "w2v2_aasist",
            "windows": windows,
            "aggregation": agg.to_dict(),
            "full_result": {
                "label": label_full, "label_str": "FAKE" if label_full == 1 else "REAL",
                "confidence": round(float(max(p_real_full, p_fake_full)), 6),
                "real_probability": round(float(p_real_full), 6),
                "fake_probability": round(float(p_fake_full), 6),
                "duration": round(total_duration, 2),
                "model_version": "w2v2_aasist",
            },
            "verdict": agg.aggregated_label_str,
            "risk_score": agg.risk_score,
            "risk_level": agg.risk_level.value,
            "real_probability": agg.final_real_prob,
            "fake_probability": agg.final_fake_prob,
            "feature_count": 64600,
        }

    from ai_engine.inference.pipeline import detect_windows

    result = detect_windows(
        audio_path,
        model_version=model_version,
        window_sec=window_sec,
        hop_sec=hop_sec,
        aggregation_method=aggregation_method,
    )
    return result.to_dict()


# ---------------------------------------------------------------------------
# Routes
# ---------------------------------------------------------------------------


@router.post("/analyze")
async def analyze_audio(
    file: UploadFile = File(..., description="Audio file (WAV/FLAC/WebM/OGG/MP3)."),
    model: str = Query("v2_robust", description="Model version: w2v2_aasist | v2_robust | v2 | v1"),
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

        # ── Validate model ────────────────────────────────
        if model not in ("w2v2_aasist", "v2_robust", "v2", "v1"):
            raise HTTPException(status_code=400, detail="model must be one of w2v2_aasist, v2_robust, v2, v1")

        # ── Run inference in thread executor ───────────────
        result = await asyncio.get_event_loop().run_in_executor(
            None, _run_detection_sync, detect_path, model
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


@router.post("/analyze-windowed")
async def analyze_windowed(
    file: UploadFile = File(..., description="Audio file (WAV/FLAC/WebM/OGG/MP3)."),
    window_sec: float = Query(4.0, ge=1.0, le=10.0, description="Window length in seconds"),
    hop_sec: float | None = Query(None, ge=0.5, le=10.0, description="Hop/stride in seconds (None = non-overlapping)"),
    aggregation: str = Query("mean", description="Aggregation method: mean|median|max|vote"),
    model: str = Query("v2_robust", description="Model version: w2v2_aasist | v2_robust | v2 | v1"),
):
    """
    Windowed detection — splits audio into 3–4 s windows and returns
    per-window P(FAKE) plus temporal aggregation.

    This is the real-time architecture endpoint.  The existing
    POST /api/detection/analyze (single-shot) is preserved unchanged.

    Query params:
    - window_sec: window length (default 4.0)
    - hop_sec: hop length (default None → non-overlapping)
    - aggregation: mean | median | max | vote
    """
    if aggregation not in ("mean", "median", "max", "vote"):
        raise HTTPException(status_code=400, detail="aggregation must be one of mean, median, max, vote")
    if model not in ("w2v2_aasist", "v2_robust", "v2", "v1"):
        raise HTTPException(status_code=400, detail="model must be one of w2v2_aasist, v2_robust, v2, v1")

    if not file.filename:
        raise HTTPException(status_code=400, detail="No filename provided.")

    ext = Path(file.filename).suffix.lower()
    if ext not in SUPPORTED_AUDIO_EXTENSIONS:
        raise HTTPException(
            status_code=400,
            detail=f"Unsupported format '{ext}'. Supported: {', '.join(sorted(SUPPORTED_AUDIO_EXTENSIONS))}",
        )

    contents = await file.read()
    if not contents:
        raise HTTPException(status_code=400, detail="Empty file.")

    max_bytes = 50 * 1024 * 1024
    if len(contents) > max_bytes:
        raise HTTPException(status_code=413, detail="File too large (max 50 MB).")

    upload_tmp = None
    converted_tmp_path = None

    try:
        upload_tmp = tempfile.NamedTemporaryFile(
            suffix=ext, delete=False, dir=str(PROJECT_ROOT / "data" / "interim")
        )
        upload_tmp.write(contents)
        upload_tmp.flush()
        upload_tmp.close()

        if ext in _NEEDS_CONVERSION:
            log.info("Converting %s → WAV via FFmpeg (windowed)", ext)
            try:
                converted_tmp_path = _convert_to_wav(upload_tmp.name, ext)
            except RuntimeError as exc:
                log.exception("FFmpeg conversion failed (windowed)")
                raise HTTPException(status_code=422, detail=f"Could not decode audio: {exc}") from exc
            detect_path = converted_tmp_path
        else:
            detect_path = upload_tmp.name

        result = await asyncio.get_event_loop().run_in_executor(
            None,
            _run_windowed_detection_sync,
            detect_path,
            window_sec,
            hop_sec,
            aggregation,
            model,
        )
        return result

    except HTTPException:
        raise
    except Exception as exc:
        log.exception("Windowed detection failed")
        raise HTTPException(status_code=500, detail=f"Windowed detection failed: {exc}") from exc
    finally:
        for p in (upload_tmp, converted_tmp_path):
            if p is not None:
                try:
                    Path(p.name if hasattr(p, "name") else p).unlink(missing_ok=True)
                except OSError:
                    pass


# ---------------------------------------------------------------------------
# Live streaming detection (WebSocket)
# ---------------------------------------------------------------------------
# Contract matches frontend/src/types/websocket.ts.
#
# Client → server:
#   {"type": "audio_chunk", "payload": {"data": "<base64 float32 LE PCM mono>",
#                                       "sample_rate": <int>, "timestamp": <ms>}}
#
#   Audio is 32-bit float PCM (full precision) — the V2 model is sensitive
#   to 16-bit quantization, so the wire format matches what librosa.load
#   produces for the feature extractor.
#
# Server → client:
#   {"type": "status_update",   "payload": {"status": "connected"|"analyzing"|"idle"}}
#   {"type": "detection_result","payload": {"label", "label_str", "confidence",
#                                           "real_probability", "fake_probability",
#                                           "chunk_duration"}}
#   {"type": "error",           "payload": {"code": <str>, "message": <str>}}

_VALID_SAMPLE_RATES = (8000, 11025, 16000, 22050, 32000, 44100, 48000)


def _decode_pcm_base64(data_b64: str) -> np.ndarray:
    """Decode base64-encoded raw float32 little-endian PCM (mono)."""
    try:
        raw = base64.b64decode(data_b64, validate=True)
    except (ValueError, TypeError) as exc:
        raise ValueError("Invalid base64 audio payload") from exc

    if len(raw) == 0:
        raise ValueError("Empty audio chunk")
    if len(raw) % 4 != 0:
        raise ValueError(
            "Expected mono float32 PCM (byte length must be a multiple of 4)"
        )

    samples = np.frombuffer(raw, dtype="<f4")
    return np.clip(samples, -1.0, 1.0).astype(np.float32)


def _detection_result_message(result) -> dict:
    """Shape a DetectionResult into the WebSocket `detection_result` payload."""
    return {
        "label": result.label,
        "label_str": result.label_str,
        "confidence": round(result.confidence, 6),
        "real_probability": round(result.real_probability, 6),
        "fake_probability": round(result.fake_probability, 6),
        "chunk_duration": round(result.duration, 3),
    }


def _run_detection_sync_array(window: np.ndarray) -> DetectionResult:
    """Run W2V2-AASIST on a live window — called in a thread executor.

    Uses the process-level cached singleton (1.2GB) — never reload per window.
    Keeps window/buffer/WS/aggregation unchanged; only the ML backend changes
    from XGBoost v2 → W2V2-AASIST. Returned DetectionResult shape and
    fake_probability semantics are identical so temporal/alert layer is untouched.
    """
    from ai_engine.inference.w2v2_aasist import TARGET_SR, get_w2v2_aasist

    model = get_w2v2_aasist()
    p_real, p_fake = model.predict_proba(window, TARGET_SR)
    label = 1 if p_fake >= 0.5 else 0
    confidence = float(max(p_real, p_fake))
    duration = float(len(window) / TARGET_SR) if len(window) else 0.0
    log.info("[LIVE DETECTION] model=w2v2_aasist p_fake=%.4f label=%s duration=%.2f", p_fake, "FAKE" if label else "REAL", duration)
    return DetectionResult(
        label=label,
        confidence=confidence,
        real_probability=float(p_real),
        fake_probability=float(p_fake),
        duration=duration,
        model_version="w2v2_aasist",
    )


@router.websocket("/ws")
async def detection_ws(websocket: WebSocket) -> None:
    """Streaming detection WebSocket — one connection = one call stream."""
    await websocket.accept()
    buffer = AudioWindowBuffer(
        window_seconds=STREAM_WINDOW_SECONDS,
        hop_seconds=STREAM_HOP_SECONDS,
        target_sr=16000,
    )
    loop = asyncio.get_event_loop()

    await websocket.send_json(
        {"type": "status_update", "payload": {"status": "connected"}}
    )
    await websocket.send_json(
        {"type": "status_update", "payload": {"status": "idle"}}
    )
    log.info("WS stream connected (window=%.1fs hop=%.1fs)", STREAM_WINDOW_SECONDS, STREAM_HOP_SECONDS)

    try:
        while True:
            message = await websocket.receive_text()
            try:
                msg = json.loads(message)
            except json.JSONDecodeError as exc:
                log.warning("Bad WS message: %s", exc)
                await websocket.send_json(
                    {"type": "error", "payload": {"code": "BAD_MESSAGE", "message": "Message is not valid JSON"}}
                )
                continue

            if msg.get("type") != "audio_chunk":
                continue

            payload = msg.get("payload") or {}
            try:
                sample_rate = int(payload["sample_rate"])
                data_b64 = payload["data"]
            except (KeyError, TypeError, ValueError):
                await websocket.send_json(
                    {"type": "error", "payload": {"code": "BAD_CHUNK", "message": "audio_chunk missing data/sample_rate"}}
                )
                continue

            if sample_rate not in _VALID_SAMPLE_RATES:
                await websocket.send_json(
                    {"type": "error", "payload": {"code": "BAD_SAMPLE_RATE", "message": f"Unsupported sample_rate {sample_rate}"}}
                )
                continue

            if sample_rate != 16000:
                # Workable but a known sensitivity: whole-file soxr
                # resampling is what training saw; browser-side per-chunk
                # resampling should be avoided by capturing natively at
                # 16 kHz (see useWebRTC.ts).
                log.warning(
                    "WS stream at %d Hz — backend will resample to 16 kHz. "
                    "Prefer capturing at 16 kHz.",
                    sample_rate,
                )

            try:
                samples = _decode_pcm_base64(data_b64)
            except ValueError as exc:
                await websocket.send_json(
                    {"type": "error", "payload": {"code": "BAD_AUDIO", "message": str(exc)}}
                )
                continue

            await websocket.send_json(
                {"type": "status_update", "payload": {"status": "analyzing"}}
            )

            windows = buffer.add_chunk(samples, sample_rate)
            for window in windows:
                # Heavy feature extraction runs in a thread — never block the loop.
                result = await loop.run_in_executor(None, _run_detection_sync_array, window)
                await websocket.send_json(
                    {"type": "detection_result", "payload": _detection_result_message(result)}
                )
                log.info(
                    "WS window done: %s (%.1fs, p_fake=%.2f)",
                    result.label_str, result.duration, result.fake_probability,
                )

    except WebSocketDisconnect:
        log.info("WS stream disconnected")
    except Exception as exc:  # noqa: BLE001 — surface any stream failure
        log.exception("WS stream error")
        try:
            await websocket.send_json(
                {"type": "error", "payload": {"code": "DETECTION_ERROR", "message": str(exc)}}
            )
        except Exception:
            pass
