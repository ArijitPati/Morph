"""
Morph Backend — Real-time streaming via WebSocket.

Architecture:
    Browser/WebRTC PCM chunks (base64) → StreamingBuffer (4 s windows)
    → 132 feature extraction → Morph V2 Robust XGBoost → P(FAKE)
    → per-window detection_result + temporal aggregation update

Protocol (JSON over WS):
    Client → Server:
        { "type": "config", "payload": { "window_sec": 4.0, "hop_sec": 4.0, "aggregation": "mean" } }
        { "type": "audio_chunk", "payload": { "data": "<base64 PCM f32>", "sample_rate": 16000, "encoding": "pcm_f32"|"pcm_s16", "timestamp": 123456 } }
        { "type": "flush", "payload": {} }   // end-of-stream, emit partial tail

    Server → Client:
        { "type": "status_update", "payload": { "status": "connected"|"analyzing"|"idle", "message": "..." } }
        { "type": "detection_result", "payload": { window_index, start_sec, end_sec, duration, label, label_str, real_probability, fake_probability, confidence, ... } }
        { "type": "aggregation_update", "payload": { AggregatedRisk } }
        { "type": "error", "payload": { "code": "...", "message": "..." } }

PCM format:
    Preferred: Float32 little-endian mono at 16 kHz (what AudioWorklet produces).
    Also accepts Int16 (browser MediaRecorder fallback).  Encoding hint optional
    — server will try f32 then s16 if not specified.

Model is loaded once at startup (v2_robust) and reused.
"""

from __future__ import annotations

import asyncio
import base64
import json
import logging
from typing import Any

import numpy as np
from fastapi import APIRouter, WebSocket, WebSocketDisconnect

log = logging.getLogger("morph-backend.realtime")

router = APIRouter(tags=["realtime"])

# We will load model lazily and use pipeline helpers directly
# to avoid coupling to main.py's _model global.


def _decode_pcm_chunk(b64_data: str, encoding: str | None, sample_rate: int) -> np.ndarray:
    """
    Decode base64 PCM chunk to float32 mono array at native rate.

    Supports:
    - pcm_f32 : Float32 LE (4 bytes/sample)
    - pcm_s16 : Int16 LE (2 bytes/sample) → converted to float32 [-1, 1]
    - auto    : try f32, then s16 (heuristic based on buffer length)
    """
    raw = base64.b64decode(b64_data)

    if encoding == "pcm_f32":
        arr = np.frombuffer(raw, dtype=np.float32)
        return arr.astype(np.float32)
    if encoding == "pcm_s16":
        arr = np.frombuffer(raw, dtype=np.int16)
        return (arr.astype(np.float32) / 32768.0)

    # Auto-detect
    # If byte length % 4 == 0 we try float32 first; heuristic: check if values
    # look plausible for float32 (abs < 2.0 for most). Otherwise treat as s16.
    if len(raw) % 4 == 0 and len(raw) >= 4:
        try:
            f32 = np.frombuffer(raw, dtype=np.float32)
            # Heuristic: if >90% of values are in [-1.5, 1.5], treat as f32
            plausible = np.mean(np.abs(f32) <= 1.5) if f32.size else 0
            if plausible > 0.9:
                return f32.astype(np.float32)
        except Exception:
            pass
    # Fallback s16
    if len(raw) % 2 == 0:
        arr = np.frombuffer(raw, dtype=np.int16)
        return (arr.astype(np.float32) / 32768.0)
    # Last resort: float32
    arr = np.frombuffer(raw, dtype=np.float32)
    return arr.astype(np.float32)


async def _run_window_inference(
    chunk: np.ndarray,
    meta,  # WindowSlice
    model_version: str,
    sr: int,
) -> dict[str, Any]:
    """
    Run 132-feature + XGBoost on one window in thread executor.
    Returns WindowResult dict.
    """
    from ai_engine.features.extraction import SR, extract_features
    from ai_engine.inference.aggregation import aggregate
    from ai_engine.inference.model import MorphModel  # noqa: used via cache below

    # Import lazily to reuse cache from pipeline
    from ai_engine.inference.pipeline import _get_model, _validate_features

    def _sync() -> dict[str, Any]:
        model = _get_model(model_version)
        features = extract_features(chunk, sr, meta.duration_sec)
        _validate_features(features, model.feature_names)
        label = model.predict(features)
        real_prob, fake_prob = model.predict_proba(features)
        conf = real_prob if label == 0 else fake_prob
        return {
            "window_index": meta.index,
            "window": meta.index + 1,
            "start_sec": meta.start_sec,
            "end_sec": meta.end_sec,
            "duration": meta.duration_sec,
            "chunk_duration": meta.duration_sec,
            "is_partial": meta.is_partial,
            "label": label,
            "label_str": "REAL" if label == 0 else "FAKE",
            "confidence": round(float(conf), 6),
            "real_probability": round(float(real_prob), 6),
            "fake_probability": round(float(fake_prob), 6),
            "real_prob": round(float(real_prob), 6),
            "fake_prob": round(float(fake_prob), 6),
            "model_version": model_version,
        }

    loop = asyncio.get_event_loop()
    return await loop.run_in_executor(None, _sync)


@router.websocket("/ws/detect")
async def websocket_detect(ws: WebSocket):
    """
    Real-time detection WebSocket.

    Query params (optional): window_sec, hop_sec, aggregation
    Can also be configured via `config` message after connect.
    """
    # Negotiate config from query string
    qp = ws.query_params
    try:
        window_sec = float(qp.get("window_sec", "4.0"))
    except ValueError:
        window_sec = 4.0
    try:
        hop_raw = qp.get("hop_sec")
        hop_sec = float(hop_raw) if hop_raw is not None else window_sec
    except ValueError:
        hop_sec = window_sec
    aggregation = qp.get("aggregation", "mean")
    if aggregation not in ("mean", "median", "max", "vote"):
        aggregation = "mean"

    model_version = "v2_robust"
    target_sr = 16000

    await ws.accept()
    log.info("WS connected — window=%.1fs hop=%.1fs agg=%s", window_sec, hop_sec, aggregation)

    # Per-connection state
    from ai_engine.inference.windowing import StreamingBuffer
    from ai_engine.inference.aggregation import aggregate

    buffer = StreamingBuffer(sr=target_sr, window_sec=window_sec, hop_sec=hop_sec)
    window_results: list[dict[str, Any]] = []
    fake_probs: list[float] = []

    async def send(obj: dict[str, Any]) -> None:
        await ws.send_text(json.dumps(obj))

    # Initial status
    await send({"type": "status_update", "payload": {"status": "connected", "message": f"Ready — window {window_sec}s hop {hop_sec}s"}})

    try:
        while True:
            raw_text = await ws.receive_text()
            try:
                msg = json.loads(raw_text)
            except json.JSONDecodeError:
                await send({"type": "error", "payload": {"code": "bad_json", "message": "Invalid JSON"}})
                continue

            mtype = msg.get("type")

            if mtype == "config":
                payload = msg.get("payload") or {}
                if "window_sec" in payload:
                    try:
                        window_sec = float(payload["window_sec"])
                        buffer.window_sec = window_sec
                        buffer.window_samples = int(round(window_sec * target_sr))
                    except Exception:
                        pass
                if "hop_sec" in payload:
                    try:
                        hop_sec = float(payload["hop_sec"])
                        buffer.hop_sec = hop_sec
                        buffer.hop_samples = int(round(hop_sec * target_sr))
                    except Exception:
                        pass
                if "aggregation" in payload and payload["aggregation"] in ("mean", "median", "max", "vote"):
                    aggregation = payload["aggregation"]
                await send({"type": "status_update", "payload": {"status": "idle", "message": f"Config updated — window {window_sec}s hop {hop_sec}s agg {aggregation}"}})
                continue

            if mtype == "audio_chunk":
                payload = msg.get("payload") or {}
                b64 = payload.get("data")
                if not b64:
                    await send({"type": "error", "payload": {"code": "missing_data", "message": "audio_chunk missing 'data' field"}})
                    continue
                sr_in = int(payload.get("sample_rate") or payload.get("sampleRate") or target_sr)
                encoding = payload.get("encoding")

                try:
                    pcm = _decode_pcm_chunk(b64, encoding, sr_in)
                except Exception as e:
                    log.warning("PCM decode failed: %s", e)
                    await send({"type": "error", "payload": {"code": "decode_error", "message": str(e)}})
                    continue

                # Resample if needed (simple: if sr_in != target_sr, resample via librosa)
                if sr_in != target_sr and pcm.size > 0:
                    try:
                        import librosa
                        pcm = librosa.resample(pcm, orig_sr=sr_in, target_sr=target_sr)
                        pcm = pcm.astype(np.float32)
                    except Exception as e:
                        log.warning("Resample failed (%s → %s): %s", sr_in, target_sr, e)
                        # proceed without resampling — feature extraction will still resample via librosa.load path,
                        # but here we have raw pcm, so warn
                        pass

                buffer.append(pcm)

                # Emit all ready windows
                while buffer.has_window():
                    nxt = buffer.next_window()
                    if nxt is None:
                        break
                    chunk, meta = nxt
                    # Run inference
                    result = await _run_window_inference(chunk, meta, model_version, target_sr)
                    window_results.append(result)
                    fake_probs.append(float(result["fake_probability"]))

                    await send({"type": "detection_result", "payload": result})

                    # Aggregation update
                    try:
                        agg = aggregate(fake_probs, method=aggregation)  # type: ignore[arg-type]
                        await send({"type": "aggregation_update", "payload": agg.to_dict()})
                    except Exception as e:
                        log.warning("Aggregation failed: %s", e)

                # Optionally send buffering status if not enough for window yet
                # (throttled — avoid spamming)
                continue

            if mtype == "flush":
                # End-of-stream — emit partial tail if any
                if buffer.has_partial():
                    nxt = buffer.flush_partial()
                    if nxt is not None:
                        chunk, meta = nxt
                        result = await _run_window_inference(chunk, meta, model_version, target_sr)
                        window_results.append(result)
                        fake_probs.append(float(result["fake_probability"]))
                        await send({"type": "detection_result", "payload": result})
                        try:
                            agg = aggregate(fake_probs, method=aggregation)  # type: ignore[arg-type]
                            await send({"type": "aggregation_update", "payload": agg.to_dict()})
                        except Exception:
                            pass
                await send({"type": "status_update", "payload": {"status": "idle", "message": "Flush complete"}})
                continue

            if mtype == "ping":
                await send({"type": "status_update", "payload": {"status": "idle", "message": "pong"}})
                continue

            # Unknown type
            await send({"type": "error", "payload": {"code": "unknown_type", "message": f"Unknown message type '{mtype}'"}})

    except WebSocketDisconnect:
        log.info("WS disconnected — processed %d windows", len(window_results))
    except Exception as e:
        log.exception("WS error: %s", e)
        try:
            await send({"type": "error", "payload": {"code": "server_error", "message": str(e)}})
        except Exception:
            pass
        try:
            await ws.close()
        except Exception:
            pass
