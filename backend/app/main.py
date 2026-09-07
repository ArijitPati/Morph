"""
Morph Backend — FastAPI application.

Loads the Morph V2 model once at startup and exposes REST endpoints
for audio detection.
"""

from __future__ import annotations

import logging
import sys
from contextlib import asynccontextmanager
from pathlib import Path

import uvicorn
from fastapi import FastAPI
from fastapi.middleware.cors import CORSMiddleware

# ---------------------------------------------------------------------------
# Path setup — make ai_engine importable
# ---------------------------------------------------------------------------

_BACKEND_DIR = Path(__file__).resolve().parent
_PROJECT_ROOT = _BACKEND_DIR.parent.parent
if str(_PROJECT_ROOT) not in sys.path:
    sys.path.insert(0, str(_PROJECT_ROOT))

from app.config import (
    CORS_ORIGINS,
    DEFAULT_MODEL_VERSION,
    PROJECT_ROOT,
    SUPPORTED_AUDIO_EXTENSIONS,
)
from app.routes.detection import router as detection_router
from app.schemas import HealthResponse

# ---------------------------------------------------------------------------
# Logging
# ---------------------------------------------------------------------------

logging.basicConfig(
    level=logging.INFO,
    format="%(asctime)s [%(levelname)s] %(message)s",
    datefmt="%H:%M:%S",
)
log = logging.getLogger("morph-backend")

# ---------------------------------------------------------------------------
# Model state — loaded once at startup
# ---------------------------------------------------------------------------

_model = None


@asynccontextmanager
async def lifespan(app: FastAPI):
    """Load the Morph model once on startup."""
    global _model
    log.info("Loading Morph %s model …", DEFAULT_MODEL_VERSION.upper())
    try:
        from ai_engine.inference.model import MorphModel

        _model = MorphModel(version=DEFAULT_MODEL_VERSION)
        log.info(
            "Model loaded — %d features, version %s",
            _model.feature_count,
            _model.version,
        )
    except Exception:
        log.exception("Failed to load model")
        _model = None
    yield
    log.info("Shutting down.")


# ---------------------------------------------------------------------------
# App
# ---------------------------------------------------------------------------

app = FastAPI(
    title="Morph Voice Detection API",
    version="0.1.0",
    description="REST API for Morph synthetic voice detection.",
    lifespan=lifespan,
)

app.add_middleware(
    CORSMiddleware,
    allow_origins=CORS_ORIGINS,
    allow_credentials=True,
    allow_methods=["*"],
    allow_headers=["*"],
)

# ---------------------------------------------------------------------------
# Routes
# ---------------------------------------------------------------------------

app.include_router(detection_router)


@app.get("/api/health", response_model=HealthResponse, tags=["health"])
async def health():
    """Health check — reports backend and AI engine status."""
    model_status = "loaded" if _model is not None else "not loaded"
    return HealthResponse(
        status="ok",
        engine=model_status,
        model_version=DEFAULT_MODEL_VERSION,
        feature_count=_model.feature_count if _model else 0,
    )


# ---------------------------------------------------------------------------
# Expose _model for detection routes
# ---------------------------------------------------------------------------


def get_loaded_model():
    """Return the model loaded at startup, or None."""
    return _model


# ---------------------------------------------------------------------------
# CLI entrypoint
# ---------------------------------------------------------------------------

if __name__ == "__main__":
    uvicorn.run(
        "app.main:app",
        host="0.0.0.0",
        port=8000,
        reload=False,
        log_level="info",
    )
