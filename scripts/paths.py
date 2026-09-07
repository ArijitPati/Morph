# scripts/paths.py
"""
Shared path utilities. Import PROJECT_ROOT and resolve_path()
in every script instead of hardcoding paths, so audio_path
values are relative-to-project-root everywhere, permanently.
"""

from pathlib import Path

# Assumes this file lives at morph/scripts/paths.py — adjust the
# number of .parent calls if you ever move it.
PROJECT_ROOT = Path(__file__).resolve().parent.parent

def to_relative(abs_path) -> str:
    """Convert an absolute path to a project-root-relative string, forward slashes."""
    rel = Path(abs_path).resolve().relative_to(PROJECT_ROOT)
    return str(rel).replace("\\", "/")

def resolve_path(rel_path: str) -> Path:
    """Convert a stored relative path back into a real, absolute Path for opening."""
    return (PROJECT_ROOT / rel_path).resolve()