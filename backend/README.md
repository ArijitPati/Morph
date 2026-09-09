# Morph Backend

FastAPI REST API for the Morph voice-clone detection system.

## Quick Start

From the project root (`morph/`):

```bash
# Ensure ai_engine is importable (run from morph/)
cd morph/
venv/bin/python -m uvicorn backend.app.main:app --host 0.0.0.0 --port 8010
```

Or from `backend/`:

```bash
cd backend/
PYTHONPATH=.. ../venv/bin/python -m uvicorn app.main:app --host 0.0.0.0 --port 8010
```

The server loads the Morph V2 XGBoost model once at startup.

## Endpoints

### GET /api/health

Health check.

```json
{
  "status": "ok",
  "engine": "loaded",
  "model_version": "v2",
  "feature_count": 132
}
```

### POST /api/detection/analyze

Upload an audio file for synthetic voice detection.

**Request:** `multipart/form-data` with a `file` field.

**Supported formats:** WAV, FLAC, WebM, OGG, MP3

**Example curl:**

```bash
curl -X POST http://localhost:8010/api/detection/analyze \
  -F "file=@path/to/audio.wav"
```

**Response:**

```json
{
  "verdict": "REAL",
  "risk_score": 2.34,
  "real_probability": 0.9766,
  "fake_probability": 0.0234,
  "model_version": "v2",
  "duration": 4.52,
  "feature_count": 132
}
```

**Error codes:**

| Code | Meaning |
|------|---------|
| 400  | Invalid format or empty file |
| 413  | File too large (>50 MB) |
| 500  | Detection pipeline error |

### WS /api/detection/ws

Live streaming detection (see `frontend/src/types/websocket.ts`). One
connection = one call stream. The client streams mono **float32 PCM**
chunks and receives a `detection_result` per completed 4 s window
(1 s hop → roughly one prediction/second).

**Client → server:**

```json
{
  "type": "audio_chunk",
  "payload": {
    "data": "<base64-encoded mono float32 LE PCM>",
    "sample_rate": 16000,
    "timestamp": 1690000000000
  }
}
```

Float32 full precision is intentional: the V2 model measurably flips some
verdicts under 16-bit quantization, so the wire format matches what
`librosa.load` feeds the feature extractor. Capture at **16000 Hz** in the
browser (`new AudioContext({ sampleRate: 16000 })`) so the backend needs
no resampling — the model is sensitive to resampler differences too.

**Server → client:**

| type | payload |
|------|---------|
| `status_update` | `{"status": "connected"\|"analyzing"\|"idle"}` |
| `detection_result` | `{"label", "label_str", "confidence", "real_probability", "fake_probability", "chunk_duration"}` |
| `error` | `{"code": "BAD_MESSAGE"\|"BAD_CHUNK"\|"BAD_SAMPLE_RATE"\|"BAD_AUDIO"\|"DETECTION_ERROR", "message": "..."}` |

## Tests

```bash
# from morph/
venv/bin/python -m pytest ai_engine/tests backend/tests -q
```
