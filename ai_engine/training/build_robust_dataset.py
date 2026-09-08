#!/usr/bin/env python3
"""
Build a robust training dataset by augmenting existing clean audio.

This script:
1. Loads existing feature Parquet files (Gary Stafford, Deep Voice, ASVspoof)
2. For each audio file, generates augmented versions simulating mic/channel effects
3. Extracts features from augmented audio using the SAME production pipeline
4. Combines original + augmented features into a single Parquet file

The augmented data teaches the model to be robust to:
- Speaker/microphone frequency response
- Room reverberation
- Background noise
- Volume/gain variation
- Resampling variation
- Codec-like degradation

Usage:
    python ai_engine/training/build_robust_dataset.py --augment-per-sample 3
    python ai_engine/training/build_robust_dataset.py --augment-per-sample 2 --limit 1000
"""

from __future__ import annotations

import argparse
import json
import logging
import sys
import time
from pathlib import Path
from typing import Any, Dict, List, Optional, Tuple

import librosa
import numpy as np
import pandas as pd
from tqdm import tqdm

# ---------------------------------------------------------------------------
# Path setup
# ---------------------------------------------------------------------------
PROJECT_ROOT = Path(__file__).resolve().parent.parent.parent
sys.path.insert(0, str(PROJECT_ROOT))

from ai_engine.features.extraction import SR, extract_features
from ai_engine.training.augmentations import AugConfig, augment_audio

# ---------------------------------------------------------------------------
# Constants
# ---------------------------------------------------------------------------
FEATURE_DIR = PROJECT_ROOT / "data/features"
OUTPUT_DIR = PROJECT_ROOT / "data/features"

METADATA_COLUMNS = {
    "sample_id", "dataset", "label", "label_name", "file_path",
    "split", "attack_id", "speaker_id",
}

logging.basicConfig(
    level=logging.INFO,
    format="%(asctime)s [%(levelname)s] %(message)s",
    datefmt="%H:%M:%S",
)
log = logging.getLogger("build_robust")


# ---------------------------------------------------------------------------
# Audio loading
# ---------------------------------------------------------------------------

def load_audio_from_path(file_path: str, sr: int = SR) -> Optional[Tuple[np.ndarray, float]]:
    """
    Load audio from a file path, handling both absolute and relative paths.
    Returns (audio_array, duration) or None if file not found.
    """
    # Try direct path first
    p = Path(file_path)
    if not p.is_absolute():
        p = PROJECT_ROOT / file_path

    if not p.exists():
        return None

    try:
        y, _ = librosa.load(str(p), sr=sr, mono=True)
        duration = librosa.get_duration(y=y, sr=sr)
        return y, duration
    except Exception:
        return None


# ---------------------------------------------------------------------------
# Augmentation + feature extraction
# ---------------------------------------------------------------------------

def augment_and_extract(
    y: np.ndarray,
    sr: int,
    duration: float,
    augment_config: AugConfig,
    rng: np.random.RandomState,
) -> Tuple[Dict[str, float], Dict[str, Any]]:
    """
    Augment audio and extract features through the production pipeline.

    Returns (features_dict, augmentation_params).
    """
    y_aug, params = augment_audio(y, sr, augment_config, rng)
    features = extract_features(y_aug, sr, duration)
    return features, params


# ---------------------------------------------------------------------------
# Dataset processing
# ---------------------------------------------------------------------------

def process_dataset(
    parquet_path: Path,
    augment_per_sample: int,
    augment_config: AugConfig,
    output_path: Path,
    limit: Optional[int] = None,
) -> None:
    """
    Load a dataset Parquet, augment each sample, extract features, and save.

    The augmented samples preserve all original metadata but add:
    - aug_id: augmentation index (0 = original, 1..N = augmented)
    - aug_params: JSON string of applied augmentation parameters
    """
    log.info("Loading dataset: %s", parquet_path)
    df = pd.read_parquet(parquet_path)
    log.info("  Loaded %d rows", len(df))

    if limit:
        df = df.head(limit)
        log.info("  Limited to %d rows", len(df))

    feature_cols = sorted(
        c for c in df.columns
        if c not in METADATA_COLUMNS and pd.api.types.is_numeric_dtype(df[c])
    )

    all_rows: List[Dict[str, Any]] = []
    errors = 0

    for idx, row in tqdm(df.iterrows(), total=len(df), desc="Augmenting", unit="sample"):
        # Get file path
        file_path = row.get("file_path", "")
        if not file_path:
            errors += 1
            continue

        # Load audio
        audio_data = load_audio_from_path(file_path)
        if audio_data is None:
            errors += 1
            continue

        y, duration = audio_data
        label = int(row["label"])

        # Original sample (aug_id = 0)
        orig_row = {}
        for col in df.columns:
            orig_row[col] = row[col] if col in row.index else None
        orig_row["aug_id"] = 0
        orig_row["aug_params"] = "{}"
        all_rows.append(orig_row)

        # Augmented samples
        rng = np.random.RandomState(42 + idx)
        for aug_i in range(augment_per_sample):
            try:
                aug_rng = np.random.RandomState(42 + idx * 1000 + aug_i)
                features, params = augment_and_extract(
                    y, SR, duration, augment_config, aug_rng
                )

                aug_row = {}
                # Copy metadata
                for col in df.columns:
                    aug_row[col] = row[col] if col in row.index else None

                # Override features with augmented features
                for fc in feature_cols:
                    aug_row[fc] = features[fc]

                # Add augmentation metadata
                aug_row["aug_id"] = aug_i + 1
                aug_row["aug_params"] = json.dumps(params)
                aug_row["sample_id"] = f"{row['sample_id']}_aug{aug_i+1}"

                all_rows.append(aug_row)

            except Exception as exc:
                errors += 1
                log.debug("Augmentation failed for %s: %s", file_path, exc)

    log.info("Generated %d total rows (%d original + %d augmented), %d errors",
             len(all_rows), len(df), len(all_rows) - len(df), errors)

    # Write output
    out_df = pd.DataFrame(all_rows)

    # Ensure consistent column ordering
    meta_cols = [c for c in out_df.columns if c in (METADATA_COLUMNS | {"aug_id", "aug_params"})]
    feat_cols = [c for c in out_df.columns if c not in (METADATA_COLUMNS | {"aug_id", "aug_params"})]
    out_df = out_df[sorted(meta_cols) + sorted(feat_cols)]

    output_path.parent.mkdir(parents=True, exist_ok=True)
    out_df.to_parquet(output_path, engine="pyarrow", index=False)
    log.info("Saved → %s (%d rows)", output_path, len(out_df))


# ---------------------------------------------------------------------------
# CLI
# ---------------------------------------------------------------------------

def main():
    parser = argparse.ArgumentParser(
        description="Build robust training dataset with audio augmentations.",
    )
    parser.add_argument(
        "--augment-per-sample",
        type=int,
        default=3,
        help="Number of augmented versions per original sample (default: 3).",
    )
    parser.add_argument(
        "--limit",
        type=int,
        default=None,
        help="Process only first N samples from each dataset (default: all).",
    )
    parser.add_argument(
        "--datasets",
        nargs="+",
        default=["gary_stafford", "deep_voice"],
        choices=["gary_stafford", "deep_voice", "asvspoof2019_la"],
        help="Datasets to augment (default: gary_stafford deep_voice).",
    )
    parser.add_argument(
        "--output-suffix",
        default="_robust",
        help="Suffix for output Parquet files (default: _robust).",
    )
    parser.add_argument(
        "--no-eq", action="store_true",
        help="Disable EQ augmentation.",
    )
    parser.add_argument(
        "--no-reverb", action="store_true",
        help="Disable reverberation augmentation.",
    )
    parser.add_argument(
        "--no-noise", action="store_true",
        help="Disable noise augmentation.",
    )
    parser.add_argument(
        "--no-gain", action="store_true",
        help="Disable gain augmentation.",
    )
    parser.add_argument(
        "--no-resample", action="store_true",
        help="Disable resampling jitter augmentation.",
    )
    parser.add_argument(
        "--no-codec", action="store_true",
        help="Disable codec/quantization augmentation.",
    )

    args = parser.parse_args()

    # Build augmentation config
    config = AugConfig(
        apply_eq=not args.no_eq,
        apply_reverb=not args.no_reverb,
        apply_noise=not args.no_noise,
        apply_gain=not args.no_gain,
        apply_resample_jitter=not args.no_resample,
        apply_codec=not args.no_codec,
    )

    log.info("Augmentation config:")
    log.info("  EQ:            %s", config.apply_eq)
    log.info("  Reverb:        %s", config.apply_reverb)
    log.info("  Noise:         %s", config.apply_noise)
    log.info("  Gain:          %s", config.apply_gain)
    log.info("  Resample:      %s", config.apply_resample_jitter)
    log.info("  Codec:         %s", config.apply_codec)
    log.info("  Aug/sample:    %d", args.augment_per_sample)
    log.info("  Limit:         %s", args.limit or "none")
    log.info("")

    for dataset in args.datasets:
        input_path = FEATURE_DIR / f"{dataset}_features.parquet"
        output_path = FEATURE_DIR / f"{dataset}{args.output_suffix}_features.parquet"

        if not input_path.exists():
            log.warning("Input not found: %s — skipping", input_path)
            continue

        log.info("=" * 60)
        log.info("Processing: %s", dataset)
        log.info("=" * 60)

        t0 = time.time()
        process_dataset(
            input_path,
            args.augment_per_sample,
            config,
            output_path,
            args.limit,
        )
        elapsed = time.time() - t0
        log.info("  Elapsed: %.1f seconds", elapsed)
        log.info("")


if __name__ == "__main__":
    main()
