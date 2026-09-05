"""
build_metadata.py

Unifies all three sources (ASVspoof 2019 LA, gary_stafford, DEEP-VOICE)
into a single master_metadata.csv with a consistent schema, applying
the agreed split strategy:

  - ASVspoof: official subset passthrough (train->train, dev->val, eval->test)
  - gary_stafford: stratified random split by label (no speaker metadata available)
  - DEEP-VOICE: entirely held out as split="test" (generalization probe only)

Labels are unified to the bonafide/spoof convention (ASVspoof standard).
"""

import csv
import soundfile as sf
from pathlib import Path
from sklearn.model_selection import train_test_split

# --- CONFIG ---
ASVSPOOF_MANIFEST = Path("data/metadata/asvspoof2019_la_manifest.csv")
DEEP_VOICE_MANIFEST = Path("data/metadata/deep_voice_chunks_manifest.csv")
GARY_REAL_MANIFEST = Path("data/metadata/gary_stafford_real_16k_manifest.csv")
GARY_FAKE_DIR = Path("data/raw/gary_stafford/fake")  # untouched, already 16kHz

OUTPUT_CSV = Path("data/metadata/master_metadata.csv")

GARY_SPLIT_RATIOS = {"train": 0.70, "val": 0.15, "test": 0.15}
RANDOM_SEED = 42

UNIFIED_FIELDS = [
    "audio_path", "source", "label", "split",
    "speaker_id", "system_id", "parent_file", "chunk_index",
    "sample_rate", "duration_sec",
]


def to_bonafide_spoof(label: str) -> str:
    """Normalize any source's label convention to bonafide/spoof."""
    label = label.lower()
    if label in ("bonafide", "real"):
        return "bonafide"
    if label in ("spoof", "fake"):
        return "spoof"
    raise ValueError(f"Unrecognized label: {label}")


def build_asvspoof_rows():
    rows = []
    subset_to_split = {"train": "train", "dev": "val", "eval": "test"}

    with open(ASVSPOOF_MANIFEST, "r") as f:
        for r in csv.DictReader(f):
            rows.append({
                "audio_path": r["audio_path"],
                "source": "asvspoof2019_la",
                "label": to_bonafide_spoof(r["label"]),
                "split": subset_to_split[r["subset"]],
                "speaker_id": r["speaker_id"],
                "system_id": r["system_id"],
                "parent_file": "",
                "chunk_index": "",
                "sample_rate": 16000,
                "duration_sec": "",  # not tracked in this manifest; audit_report.csv has it if needed
            })
    print(f"asvspoof2019_la: {len(rows)} rows")
    return rows


def build_deep_voice_rows():
    rows = []
    with open(DEEP_VOICE_MANIFEST, "r") as f:
        for r in csv.DictReader(f):
            rows.append({
                "audio_path": r["chunk_path"],
                "source": "deep_voice",
                "label": to_bonafide_spoof(r["label"]),
                "split": "test",  # hardcoded: DEEP-VOICE is a held-out generalization probe only
                "speaker_id": "",
                "system_id": "",
                "parent_file": r["parent_file"],
                "chunk_index": r["chunk_index"],
                "sample_rate": r["sample_rate"],
                "duration_sec": r["duration_sec"],
            })
    print(f"deep_voice: {len(rows)} rows")
    return rows


def build_gary_stafford_rows():
    raw_rows = []

    # real class: resampled, read from its manifest
    with open(GARY_REAL_MANIFEST, "r") as f:
        for r in csv.DictReader(f):
            raw_rows.append({
                "audio_path": r["resampled_path"],
                "source": "gary_stafford",
                "label": "bonafide",
                "speaker_id": "",
                "system_id": "",
                "parent_file": "",
                "chunk_index": "",
                "sample_rate": r["target_sample_rate"],
                "duration_sec": r["duration_sec"],
            })

    # fake class: untouched, already 16kHz — scan folder directly
    fake_files = sorted(GARY_FAKE_DIR.glob("*"))
    for f in fake_files:
        try:
            info = sf.info(str(f))
            duration = info.frames / info.samplerate
        except Exception as e:
            print(f"[warn] could not read {f}: {e}")
            continue
        raw_rows.append({
            "audio_path": str(f.resolve()),
            "source": "gary_stafford",
            "label": "spoof",
            "speaker_id": "",
            "system_id": "",
            "parent_file": "",
            "chunk_index": "",
            "sample_rate": info.samplerate,
            "duration_sec": duration,
        })

    print(f"gary_stafford: {len(raw_rows)} rows before split "
          f"({sum(1 for r in raw_rows if r['label']=='bonafide')} bonafide, "
          f"{sum(1 for r in raw_rows if r['label']=='spoof')} spoof)")

    # --- stratified split by label (no speaker metadata available for this source) ---
    labels = [r["label"] for r in raw_rows]

    train_rows, temp_rows = train_test_split(
        raw_rows, test_size=(1 - GARY_SPLIT_RATIOS["train"]),
        stratify=labels, random_state=RANDOM_SEED
    )
    temp_labels = [r["label"] for r in temp_rows]
    val_size = GARY_SPLIT_RATIOS["val"] / (GARY_SPLIT_RATIOS["val"] + GARY_SPLIT_RATIOS["test"])
    val_rows, test_rows = train_test_split(
        temp_rows, test_size=(1 - val_size),
        stratify=temp_labels, random_state=RANDOM_SEED
    )

    for r in train_rows:
        r["split"] = "train"
    for r in val_rows:
        r["split"] = "val"
    for r in test_rows:
        r["split"] = "test"

    all_rows = train_rows + val_rows + test_rows
    print(f"gary_stafford split: train={len(train_rows)}, val={len(val_rows)}, test={len(test_rows)}")
    return all_rows


def main():
    all_rows = []
    all_rows.extend(build_asvspoof_rows())
    all_rows.extend(build_deep_voice_rows())
    all_rows.extend(build_gary_stafford_rows())

    OUTPUT_CSV.parent.mkdir(parents=True, exist_ok=True)
    with open(OUTPUT_CSV, "w", newline="") as f:
        writer = csv.DictWriter(f, fieldnames=UNIFIED_FIELDS)
        writer.writeheader()
        writer.writerows(all_rows)

    print(f"\nWrote {len(all_rows)} total rows to {OUTPUT_CSV}")

    # --- summary breakdown ---
    from collections import Counter
    split_label_counts = Counter((r["split"], r["source"], r["label"]) for r in all_rows)
    print("\nsplit      source            label      count")
    for key in sorted(split_label_counts.keys()):
        split, source, label = key
        print(f"{split:<10} {source:<17} {label:<10} {split_label_counts[key]}")


if __name__ == "__main__":
    main()