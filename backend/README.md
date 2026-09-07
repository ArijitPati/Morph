# Morph Backend

FastAPI REST API for the Morph voice-clone detection system.

## Quick Start

From the project root (`morph/`):

```bash
# Ensure ai_engine is importable (run from morph/)
cd morph/
venv_new/bin/python -m uvicorn backend.app.main:app --host 0.0.0.0 --port 8000
```

Or from `backend/`:

```bash
cd backend/
PYTHONPATH=.. ../venv_new/bin/python -m uvicorn app.main:app --host 0.0.0.0 --port 8000
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
curl -X POST http://localhost:8000/api/detection/analyze \
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
