"""Application configuration."""

from pathlib import Path

# Project root (morph/)
PROJECT_ROOT = Path(__file__).resolve().parent.parent.parent

# CORS origins — Next.js dev server
CORS_ORIGINS = [
    "http://localhost:3000",
    "http://127.0.0.1:3000",
]

# Upload limits
MAX_UPLOAD_SIZE_MB = 50
SUPPORTED_AUDIO_EXTENSIONS = {".wav", ".flac", ".webm", ".ogg", ".mp3"}

# Model
DEFAULT_MODEL_VERSION = "v2"
