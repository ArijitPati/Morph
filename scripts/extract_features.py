#!/usr/bin/env python3
"""
Production-ready batch feature extraction for the Morph voice-clone
detection project.

Extracts acoustic / spectral / cepstral features from three datasets
and writes them to Parquet files, one per dataset.

Usage:
    python scripts/extract_features.py --dataset gary_stafford
    python scripts/extract_features.py --dataset deep_voice
    python scripts/extract_features.py --dataset asvspoof2019_la
    python scripts/extract_features.py --dataset all
    python scripts/extract_features.py --dataset all --batch-size 1000
"""

from __future__ import annotations

import argparse
import logging
import os
import sys
import time
from pathlib import Path
from typing import Any, Dict, List, Optional, Tuple

import librosa
import numpy as np
import pandas as pd
from tqdm import tqdm

# ---------------------------------------------------------------------------
# Feature extraction — imported from the reusable ai_engine module.
# The extraction logic is defined once in ai_engine.features.extraction
# and reused here (and by the inference pipeline).
# ---------------------------------------------------------------------------

sys.path.insert(0, str(Path(__file__).resolve().parent.parent))
from ai_engine.features.extraction import SR, extract_features  # noqa: E402

# ---------------------------------------------------------------------------
# Constants
# ---------------------------------------------------------------------------

# Supported audio extensions
AUDIO_EXTENSIONS = (".wav", ".flac")

# Project root (two levels up from this script)
PROJECT_ROOT = Path(__file__).resolve().parent.parent

# ASVspoof protocol directory (relative to project root)
ASVSPOOF_PROTOCOL_DIR = (
    "data/raw/asvspoof2019_la/LA/ASVspoof2019_LA_cm_protocols"
)

# ASVspoof split directories (relative to project root)
ASVSPOOF_SPLITS = {
    "train": "data/raw/asvspoof2019_la/LA/ASVspoof2019_LA_train/flac",
    "dev": "data/raw/asvspoof2019_la/LA/ASVspoof2019_LA_dev/flac",
    "eval": "data/raw/asvspoof2019_la/LA/ASVspoof2019_LA_eval/flac",
}

# Protocol file names (speaker → utterance → system → attack → label)
ASVSPOOF_PROTOCOL_FILES = {
    "train": "ASVspoof2019.LA.cm.train.trn.txt",
    "dev": "ASVspoof2019.LA.cm.dev.trl.txt",
    "eval": "ASVspoof2019.LA.cm.eval.trl.txt",
}

# Batch flush size (rows)
DEFAULT_BATCH_SIZE = 500

# Metadata columns (not features)
METADATA_COLUMNS = {"sample_id", "dataset", "label", "label_name", "file_path"}
ASVSPOOF_EXTRA_COLUMNS = {"split", "attack_id", "speaker_id"}

# ---------------------------------------------------------------------------
# Logging
# ---------------------------------------------------------------------------

logging.basicConfig(
    level=logging.INFO,
    format="%(asctime)s [%(levelname)s] %(message)s",
    datefmt="%H:%M:%S",
)
log = logging.getLogger("extract_features")


# ---------------------------------------------------------------------------
# Feature extraction helpers
# ---------------------------------------------------------------------------
# NOTE: _safe_stats, extract_cqcc, extract_pitch, and extract_features
# are now imported from ai_engine.features.extraction (see imports above).
# This file retains only the batch-processing CLI for offline dataset
# feature extraction.


# ---------------------------------------------------------------------------
# ASVspoof protocol parser
# ---------------------------------------------------------------------------

def parse_asvspoof_protocols() -> Dict[str, Dict[str, Any]]:
    """
    Parse all three ASVspoof 2019 LA CM protocol files and return a
    lookup dictionary mapping utterance_id to metadata.

    Protocol format (space-separated):
        speaker_id  utterance_id  system_id  attack_id  label

    Returns:
        {utterance_id: {"speaker_id": ..., "attack_id": ...,
                        "label": 0|1, "label_name": ..., "split": ...}}
    """
    proto_dir = PROJECT_ROOT / ASVSPOOF_PROTOCOL_DIR
    lookup: Dict[str, Dict[str, Any]] = {}

    for split, fname in ASVSPOOF_PROTOCOL_FILES.items():
        fpath = proto_dir / fname
        if not fpath.exists():
            log.warning("Protocol file not found: %s", fpath)
            continue

        with open(fpath, "r") as fh:
            for line in fh:
                line = line.strip()
                if not line:
                    continue
                parts = line.split()
                if len(parts) < 5:
                    continue

                speaker_id = parts[0]
                utterance_id = parts[1]
                # parts[2] is system_id (always "-" in CM protocols)
                attack_id = parts[3]   # "-" for bonafide
                label_str = parts[4]   # "bonafide" or "spoof"

                label = 0 if label_str == "bonafide" else 1
                label_name = "bonafide" if label == 0 else "spoof"

                lookup[utterance_id] = {
                    "speaker_id": speaker_id,
                    "attack_id": attack_id if attack_id != "-" else "",
                    "label": label,
                    "label_name": label_name,
                    "split": split,
                }

    log.info("Parsed ASVspoof protocols: %d utterances mapped", len(lookup))
    return lookup


# ---------------------------------------------------------------------------
# Dataset processors
# ---------------------------------------------------------------------------

def _collect_wav_files(folder: str) -> List[str]:
    """Recursively collect audio file paths from a directory."""
    files: List[str] = []
    for root, _, filenames in os.walk(folder):
        for fn in filenames:
            if fn.lower().endswith(AUDIO_EXTENSIONS):
                files.append(os.path.join(root, fn))
    return sorted(files)


def process_gary_stafford(
    batch_size: int,
    output_dir: Path,
    limit: Optional[int] = None,
) -> None:
    """
    Process the Gary Stafford dataset.

    Directory layout:
        data/raw/gary_stafford/real/*.wav
        data/raw/gary_stafford/fake/*.wav
    """
    dataset_name = "gary_stafford"
    log.info("=== Processing dataset: %s ===", dataset_name)

    class_map = {
        "real": (0, "real"),
        "fake": (1, "fake"),
    }

    all_files: List[Tuple[str, int, str]] = []  # (path, label, label_name)
    for class_label_name, (label, _) in class_map.items():
        folder = str(PROJECT_ROOT / f"data/raw/gary_stafford/{class_label_name}")
        if not os.path.isdir(folder):
            log.warning("Directory not found: %s", folder)
            continue
        for fp in _collect_wav_files(folder):
            all_files.append((fp, label, class_label_name))

    log.info("Found %d files for %s", len(all_files), dataset_name)
    if limit is not None:
        all_files = all_files[:limit]
        log.info("Limited to first %d files", len(all_files))
    _process_and_write(all_files, dataset_name, batch_size, output_dir)


def process_deep_voice(
    batch_size: int,
    output_dir: Path,
    limit: Optional[int] = None,
) -> None:
    """
    Process the Deep Voice dataset (chunked).

    Directory layout:
        data/interim/deep_voice_chunks/real/*.wav
        data/interim/deep_voice_chunks/fake/*.wav
    """
    dataset_name = "deep_voice"
    log.info("=== Processing dataset: %s ===", dataset_name)

    class_map = {
        "real": (0, "real"),
        "fake": (1, "fake"),
    }

    all_files: List[Tuple[str, int, str]] = []
    for class_label_name, (label, _) in class_map.items():
        folder = str(PROJECT_ROOT / f"data/interim/deep_voice_chunks/{class_label_name}")
        if not os.path.isdir(folder):
            log.warning("Directory not found: %s", folder)
            continue
        for fp in _collect_wav_files(folder):
            all_files.append((fp, label, class_label_name))

    log.info("Found %d files for %s", len(all_files), dataset_name)
    if limit is not None:
        all_files = all_files[:limit]
        log.info("Limited to first %d files", len(all_files))
    _process_and_write(all_files, dataset_name, batch_size, output_dir)


def process_asvspoof(
    batch_size: int,
    output_dir: Path,
    limit: Optional[int] = None,
) -> None:
    """
    Process the ASVspoof 2019 LA dataset.

    Audio lives under split-specific flac/ subdirectories.  Labels and
    split assignments come exclusively from the CM protocol files.
    """
    dataset_name = "asvspoof2019_la"
    log.info("=== Processing dataset: %s ===", dataset_name)

    # 1. Parse protocols
    protocol_lookup = parse_asvspoof_protocols()
    if not protocol_lookup:
        log.error("No protocol entries found — aborting ASVspoof processing.")
        return

    # 2. Collect audio files and map through protocols
    all_files: List[Tuple[str, int, str, Dict[str, Any]]] = []

    for split_name, split_rel_dir in ASVSPOOF_SPLITS.items():
        split_dir = PROJECT_ROOT / split_rel_dir
        if not split_dir.is_dir():
            log.warning("Split directory not found: %s", split_dir)
            continue

        for fp in _collect_wav_files(str(split_dir)):
            # Derive utterance_id from filename (strip extension)
            utterance_id = Path(fp).stem

            if utterance_id not in protocol_lookup:
                # Skip files not in the protocol
                continue

            meta = protocol_lookup[utterance_id]
            all_files.append((fp, meta["label"], meta["label_name"], meta))

    log.info(
        "Mapped %d ASVspoof files through protocols (%d total in lookup)",
        len(all_files),
        len(protocol_lookup),
    )
    if limit is not None:
        all_files = all_files[:limit]
        log.info("Limited to first %d files", len(all_files))
    _process_and_write_asvspoof(all_files, dataset_name, batch_size, output_dir)


# ---------------------------------------------------------------------------
# Generic batch processing + Parquet writing
# ---------------------------------------------------------------------------

def _flush_to_parquet(
    rows: List[Dict[str, Any]],
    output_path: Path,
    append: bool = False,
) -> None:
    """Write a list of row-dicts to a Parquet file."""
    if not rows:
        return
    df = pd.DataFrame(rows)

    # Ensure consistent column ordering: metadata first, then features
    meta_cols = [c for c in df.columns if c in (METADATA_COLUMNS | ASVSPOOF_EXTRA_COLUMNS)]
    feat_cols = [c for c in df.columns if c not in (METADATA_COLUMNS | ASVSPOOF_EXTRA_COLUMNS)]
    df = df[sorted(meta_cols) + sorted(feat_cols)]

    if append and output_path.exists():
        existing = pd.read_parquet(output_path)
        df = pd.concat([existing, df], ignore_index=True)

    df.to_parquet(output_path, engine="pyarrow", index=False)


def _process_and_write(
    all_files: List[Tuple[str, int, str]],
    dataset_name: str,
    batch_size: int,
    output_dir: Path,
) -> None:
    """
    Generic processor for gary_stafford and deep_voice.

    all_files: list of (file_path, label, label_name)
    """
    output_path = output_dir / f"{dataset_name}_features.parquet"
    error_log_path = output_dir / f"{dataset_name}_errors.log"

    # If output already exists, create a numbered backup name
    if output_path.exists():
        ts = int(time.time())
        backup = output_dir / f"{dataset_name}_features_{ts}.parquet"
        log.warning("Output file exists — renaming to %s", backup.name)
        output_path.rename(backup)

    total = len(all_files)
    processed = 0
    errors = 0
    batch_rows: List[Dict[str, Any]] = []

    log.info("Processing %d files …", total)

    with open(error_log_path, "w") as err_fh:
        for fp, label, label_name in tqdm(all_files, desc=dataset_name, unit="file"):
            try:
                y, sr = librosa.load(fp, sr=SR, mono=True)
                duration = librosa.get_duration(y=y, sr=sr)
                feats = extract_features(y, sr, duration)

                # Metadata
                feats["sample_id"] = Path(fp).stem
                feats["dataset"] = dataset_name
                feats["label"] = label
                feats["label_name"] = label_name
                feats["file_path"] = str(Path(fp).resolve().relative_to(PROJECT_ROOT))

                batch_rows.append(feats)
                processed += 1

            except Exception as exc:
                errors += 1
                err_fh.write(f"{fp}\t{exc}\n")

            # Flush batch
            if len(batch_rows) >= batch_size:
                _flush_to_parquet(batch_rows, output_path, append=True)
                log.info(
                    "  Flushed batch — %d/%d processed, %d errors",
                    processed, total, errors,
                )
                batch_rows.clear()

    # Final flush
    if batch_rows:
        _flush_to_parquet(batch_rows, output_path, append=True)

    log.info(
        "DONE %s: %d processed, %d errors → %s",
        dataset_name, processed, errors, output_path,
    )


def _process_and_write_asvspoof(
    all_files: List[Tuple[str, int, str, Dict[str, Any]]],
    dataset_name: str,
    batch_size: int,
    output_dir: Path,
) -> None:
    """
    ASVspoof-specific processor that carries extra metadata columns
    (split, attack_id, speaker_id).
    """
    output_path = output_dir / f"{dataset_name}_features.parquet"
    error_log_path = output_dir / f"{dataset_name}_errors.log"

    if output_path.exists():
        ts = int(time.time())
        backup = output_dir / f"{dataset_name}_features_{ts}.parquet"
        log.warning("Output file exists — renaming to %s", backup.name)
        output_path.rename(backup)

    total = len(all_files)
    processed = 0
    errors = 0
    batch_rows: List[Dict[str, Any]] = []

    log.info("Processing %d files …", total)

    with open(error_log_path, "w") as err_fh:
        for fp, label, label_name, meta in tqdm(all_files, desc=dataset_name, unit="file"):
            try:
                y, sr = librosa.load(fp, sr=SR, mono=True)
                duration = librosa.get_duration(y=y, sr=sr)
                feats = extract_features(y, sr, duration)

                # Metadata
                feats["sample_id"] = Path(fp).stem
                feats["dataset"] = dataset_name
                feats["label"] = label
                feats["label_name"] = label_name
                feats["file_path"] = str(Path(fp).resolve().relative_to(PROJECT_ROOT))

                # ASVspoof-specific metadata
                feats["split"] = meta["split"]
                feats["attack_id"] = meta["attack_id"]
                feats["speaker_id"] = meta["speaker_id"]

                batch_rows.append(feats)
                processed += 1

            except Exception as exc:
                errors += 1
                err_fh.write(f"{fp}\t{exc}\n")

            if len(batch_rows) >= batch_size:
                _flush_to_parquet(batch_rows, output_path, append=True)
                log.info(
                    "  Flushed batch — %d/%d processed, %d errors",
                    processed, total, errors,
                )
                batch_rows.clear()

    if batch_rows:
        _flush_to_parquet(batch_rows, output_path, append=True)

    log.info(
        "DONE %s: %d processed, %d errors → %s",
        dataset_name, processed, errors, output_path,
    )


# ---------------------------------------------------------------------------
# CLI
# ---------------------------------------------------------------------------

def build_parser() -> argparse.ArgumentParser:
    parser = argparse.ArgumentParser(
        description="Extract acoustic features from Morph datasets.",
    )
    parser.add_argument(
        "--dataset",
        required=True,
        choices=["gary_stafford", "deep_voice", "asvspoof2019_la", "all"],
        help="Which dataset to process (or 'all').",
    )
    parser.add_argument(
        "--output-dir",
        default=str(PROJECT_ROOT / "data/features"),
        help="Directory for output Parquet files (default: data/features).",
    )
    parser.add_argument(
        "--batch-size",
        type=int,
        default=DEFAULT_BATCH_SIZE,
        help=f"Rows per batch flush (default: {DEFAULT_BATCH_SIZE}).",
    )
    parser.add_argument(
        "--limit",
        type=int,
        default=None,
        help="Process only the first N files (default: all).",
    )
    return parser


def main(argv: Optional[List[str]] = None) -> None:
    parser = build_parser()
    args = parser.parse_args(argv)

    output_dir = Path(args.output_dir)
    output_dir.mkdir(parents=True, exist_ok=True)

    datasets_to_process: List[str]
    if args.dataset == "all":
        datasets_to_process = ["gary_stafford", "deep_voice", "asvspoof2019_la"]
    else:
        datasets_to_process = [args.dataset]

    log.info("Output directory : %s", output_dir)
    log.info("Batch size       : %d", args.batch_size)
    log.info("Limit            : %s", args.limit or "unlimited")
    log.info("Datasets         : %s", datasets_to_process)

    for ds in datasets_to_process:
        if ds == "gary_stafford":
            process_gary_stafford(args.batch_size, output_dir, args.limit)
        elif ds == "deep_voice":
            process_deep_voice(args.batch_size, output_dir, args.limit)
        elif ds == "asvspoof2019_la":
            process_asvspoof(args.batch_size, output_dir, args.limit)

    log.info("All requested datasets processed.")


if __name__ == "__main__":
    main()
