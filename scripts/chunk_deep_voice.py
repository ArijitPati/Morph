"""
chunk_deep_voice.py

Prepares DEEP-VOICE for merging with the rest of the dataset:
  1. Resample each file to 16kHz (matches ASVspoof / gary_stafford target rate)
  2. Trim leading/trailing silence using RMS-energy-based VAD (librosa.effects.trim)
  3. Split into 4-second, non-overlapping windows
  4. Discard any window that's still mostly silent, and discard trailing
     remainders shorter than 2 seconds
  5. Write chunks to data/interim/deep_voice_chunks/{real,fake}/
     and log everything to a manifest CSV (parent_file, chunk_index, label, chunk_path)
"""

import os
import csv
import librosa
import soundfile as sf
import numpy as np
from pathlib import Path
from paths import to_relative

# --- CONFIG ---
RAW_ROOT = Path("data/raw/deep_voice")
OUTPUT_ROOT = Path("data/interim/deep_voice_chunks")
MANIFEST_CSV = Path("data/metadata/deep_voice_chunks_manifest.csv")

TARGET_SR = 16000
WINDOW_SEC = 4.0
WINDOW_SAMPLES = int(WINDOW_SEC * TARGET_SR)          # 64,000
MIN_REMAINDER_SEC = 2.0
MIN_REMAINDER_SAMPLES = int(MIN_REMAINDER_SEC * TARGET_SR)

TRIM_TOP_DB = 30          # librosa.effects.trim threshold — higher = more aggressive trim
WINDOW_SILENCE_RMS_THRESHOLD = 0.01   # discard a window if its RMS falls below this

LABELS = ["real", "fake"]


def is_mostly_silent(chunk: np.ndarray, threshold: float = WINDOW_SILENCE_RMS_THRESHOLD) -> bool:
    rms = np.sqrt(np.mean(chunk ** 2))
    return rms < threshold


def process_file(filepath: Path, label: str, manifest_rows: list):
    # Step 1: load + resample to 16kHz in one call
    try:
        audio, sr = librosa.load(str(filepath), sr=TARGET_SR, mono=True)
    except Exception as e:
        print(f"[error] failed to load {filepath}: {e}")
        return

    # Step 2: RMS-energy-based silence trim (leading/trailing only)
    trimmed_audio, _ = librosa.effects.trim(audio, top_db=TRIM_TOP_DB)

    if len(trimmed_audio) < MIN_REMAINDER_SAMPLES:
        print(f"[skip] {filepath.name}: nothing but silence after trim")
        return

    # Step 3: non-overlapping 4s windowing
    n_full_windows = len(trimmed_audio) // WINDOW_SAMPLES
    remainder_samples = len(trimmed_audio) - (n_full_windows * WINDOW_SAMPLES)

    out_dir = OUTPUT_ROOT / label
    out_dir.mkdir(parents=True, exist_ok=True)

    parent_stem = filepath.stem
    chunk_index = 0

    for i in range(n_full_windows):
        start = i * WINDOW_SAMPLES
        end = start + WINDOW_SAMPLES
        chunk = trimmed_audio[start:end]

        if is_mostly_silent(chunk):
            continue  # step 4: discard windows that are still mostly silence

        chunk_filename = f"{parent_stem}_chunk{chunk_index:04d}.wav"
        chunk_path = out_dir / chunk_filename
        sf.write(str(chunk_path), chunk, TARGET_SR)

        manifest_rows.append({
            "parent_file": to_relative(filepath),
            "chunk_index": chunk_index,
            "label": label,
            "chunk_path": to_relative(chunk_path),
            "duration_sec": WINDOW_SEC,
            "sample_rate": TARGET_SR,
        })
        chunk_index += 1

    # Step 4 (remainder handling): only keep if >= MIN_REMAINDER_SEC and not silent
    if remainder_samples >= MIN_REMAINDER_SAMPLES:
        start = n_full_windows * WINDOW_SAMPLES
        chunk = trimmed_audio[start:]

        if not is_mostly_silent(chunk):
            chunk_filename = f"{parent_stem}_chunk{chunk_index:04d}.wav"
            chunk_path = out_dir / chunk_filename
            sf.write(str(chunk_path), chunk, TARGET_SR)

            manifest_rows.append({
                "parent_file": to_relative(filepath),
                "chunk_index": chunk_index,
                "label": label,
                "chunk_path": to_relative(chunk_path),
                "duration_sec": len(chunk) / TARGET_SR,
                "sample_rate": TARGET_SR,
            })


def main():
    manifest_rows = []

    for label in LABELS:
        folder = RAW_ROOT / label
        files = sorted(folder.glob("*"))
        print(f"Processing {len(files)} files in {folder}...")

        for f in files:
            process_file(f, label, manifest_rows)

    MANIFEST_CSV.parent.mkdir(parents=True, exist_ok=True)
    with open(MANIFEST_CSV, "w", newline="") as f:
        writer = csv.DictWriter(f, fieldnames=[
            "parent_file", "chunk_index", "label", "chunk_path", "duration_sec", "sample_rate"
        ])
        writer.writeheader()
        writer.writerows(manifest_rows)

    print(f"\nWrote {len(manifest_rows)} chunks to {OUTPUT_ROOT}")
    print(f"Manifest saved to {MANIFEST_CSV}")

    # quick per-label summary
    from collections import Counter
    counts = Counter(r["label"] for r in manifest_rows)
    parent_counts = {
        label: len(set(r["parent_file"] for r in manifest_rows if r["label"] == label))
        for label in LABELS
    }
    for label in LABELS:
        print(f"  {label}: {counts.get(label, 0)} chunks from {parent_counts.get(label, 0)} source files")


if __name__ == "__main__":
    main()