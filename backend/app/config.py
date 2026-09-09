"""Application configuration."""

from pathlib import Path

# Project root (morph/)
PROJECT_ROOT = Path(__file__).resolve().parent.parent.parent

# CORS origins — Next.js dev server.
# Diagnostic LAN mode: allow any origin so a second laptop on the same
# network can reach the API. tighten before any public/demo deployment.
CORS_ORIGINS = ["*"]

# Upload limits
MAX_UPLOAD_SIZE_MB = 50
SUPPORTED_AUDIO_EXTENSIONS = {".wav", ".flac", ".webm", ".ogg", ".mp3"}

# Model
DEFAULT_MODEL_VERSION = "v2_robust"
