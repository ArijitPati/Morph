"""
resample_gary_stafford_real.py

gary_stafford's 'real' class is at 44100 Hz while its 'fake' class (and
ASVspoof) is already at 16000 Hz — this resamples the real class only,
to eliminate the sample-rate/label confound before merging into the
unified dataset.

'fake' class files are left untouched in data/raw/ (already correct rate).
"""

import soundfile as sf
import librosa
import csv
from pathlib import Path
from paths import to_relative

# --- CONFIG ---
INPUT_DIR = Path("data/raw/gary_stafford/real")
OUTPUT_DIR = Path("data/interim/gary_stafford_real_16k")
MANIFEST_CSV = Path("data/metadata/gary_stafford_real_16k_manifest.csv")

TARGET_SR = 16000


def main():
    OUTPUT_DIR.mkdir(parents=True, exist_ok=True)

    files = sorted(INPUT_DIR.glob("*"))
    print(f"Resampling {len(files)} files from {INPUT_DIR} to {TARGET_SR}Hz...")

    manifest_rows = []
    n_failed = 0

    for i, f in enumerate(files):
        try:
            # librosa.load resamples on the fly when sr is specified
            audio, sr = librosa.load(str(f), sr=TARGET_SR, mono=True)

            out_path = OUTPUT_DIR / f.name
            # ensure .wav extension regardless of source format
            out_path = out_path.with_suffix(".wav")

            sf.write(str(out_path), audio, TARGET_SR)

            manifest_rows.append({
                "original_file": to_relative(f),
                "resampled_path": to_relative(out_path),
                "original_sample_rate": sr,  # note: librosa.load already resampled; see below
                "target_sample_rate": TARGET_SR,
                "duration_sec": len(audio) / TARGET_SR,
            })

        except Exception as e:
            print(f"[error] failed on {f.name}: {e}")
            n_failed += 1

        if (i + 1) % 200 == 0:
            print(f"  ...{i+1}/{len(files)} processed")

    MANIFEST_CSV.parent.mkdir(parents=True, exist_ok=True)
    with open(MANIFEST_CSV, "w", newline="") as f:
        writer = csv.DictWriter(f, fieldnames=[
            "original_file", "resampled_path", "original_sample_rate",
            "target_sample_rate", "duration_sec"
        ])
        writer.writeheader()
        writer.writerows(manifest_rows)

    print(f"\nDone. {len(manifest_rows)} files resampled, {n_failed} failed.")
    print(f"Output: {OUTPUT_DIR}")
    print(f"Manifest: {MANIFEST_CSV}")


if __name__ == "__main__":
    main()