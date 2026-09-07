"""
validate_metadata.py

Final pre-feature-extraction validation pass:
  1. Channel consistency check — confirms every source is mono (channels == 1)
     using audit_report.csv (already-computed data, no re-reading of audio).
  2. Path existence check — confirms every audio_path in master_metadata.csv
     actually resolves to a real file on disk.
  3. Fills in duration_sec for ASVspoof rows in master_metadata.csv, which
     were left blank since that manifest never tracked duration.

Does not modify audio or re-run the full audit — this is a fast sanity pass.
"""

import csv
import soundfile as sf
from pathlib import Path
from collections import defaultdict

AUDIT_REPORT_CSV = Path("data/metadata/audit_report.csv")
MASTER_METADATA_CSV = Path("data/metadata/master_metadata.csv")
OUTPUT_METADATA_CSV = Path("data/metadata/master_metadata.csv")  # overwrite in place


def check_channel_consistency():
    print("=== Channel consistency check (from audit_report.csv) ===")
    channel_counts = defaultdict(lambda: defaultdict(int))  # source -> channels -> count

    with open(AUDIT_REPORT_CSV, "r") as f:
        for row in csv.DictReader(f):
            source = row["source"]
            channels = row.get("channels", "")
            if channels == "" or channels is None:
                continue
            channel_counts[source][channels] += 1

    problems_found = False
    for source, counts in channel_counts.items():
        distinct_channel_values = list(counts.keys())
        if len(distinct_channel_values) > 1 or distinct_channel_values != ["1"]:
            problems_found = True
            print(f"[WARNING] {source}: mixed or non-mono channel counts found -> {dict(counts)}")
        else:
            print(f"[ok] {source}: all mono ({counts})")

    if not problems_found:
        print("All sources confirmed mono. No action needed.\n")
    else:
        print("!! Some sources are not uniformly mono. Force mono=True at feature-extraction load time.\n")

    return problems_found


def check_paths_and_fill_asvspoof_duration():
    print("=== Path existence check + ASVspoof duration fill (master_metadata.csv) ===")

    with open(MASTER_METADATA_CSV, "r") as f:
        reader = csv.DictReader(f)
        fieldnames = reader.fieldnames
        rows = list(reader)

    print(f"Checking {len(rows)} rows...")

    missing_paths = []
    duration_filled = 0
    duration_failed = 0

    for i, row in enumerate(rows):
        path = row["audio_path"]

        if not Path(path).exists():
            missing_paths.append((row["source"], path))
            continue

        # fill duration_sec for asvspoof rows only (left blank at build_metadata time)
        if row["source"] == "asvspoof2019_la" and (row["duration_sec"] == "" or row["duration_sec"] is None):
            try:
                info = sf.info(path)
                row["duration_sec"] = info.frames / info.samplerate
                duration_filled += 1
            except Exception as e:
                print(f"[warn] could not read duration for {path}: {e}")
                duration_failed += 1

        if (i + 1) % 20000 == 0:
            print(f"  ...{i+1}/{len(rows)} checked")

    # --- report missing paths ---
    if missing_paths:
        print(f"\n[WARNING] {len(missing_paths)} audio_path entries do not resolve to real files:")
        by_source = defaultdict(int)
        for source, path in missing_paths:
            by_source[source] += 1
        for source, count in by_source.items():
            print(f"  {source}: {count} missing")
        print("  (first 5 examples)")
        for source, path in missing_paths[:5]:
            print(f"    [{source}] {path}")
    else:
        print("\nAll audio_path entries resolved successfully. No missing files.")

    print(f"\nDuration filled for {duration_filled} ASVspoof rows ({duration_failed} failed to read).")

    # --- write updated metadata back out ---
    with open(OUTPUT_METADATA_CSV, "w", newline="") as f:
        writer = csv.DictWriter(f, fieldnames=fieldnames)
        writer.writeheader()
        writer.writerows(rows)

    print(f"Updated master_metadata.csv written to {OUTPUT_METADATA_CSV}\n")

    return missing_paths


def main():
    channel_problems = check_channel_consistency()
    missing_paths = check_paths_and_fill_asvspoof_duration()

    print("=== FINAL VERDICT ===")
    if not channel_problems and not missing_paths:
        print("Pipeline validated. Safe to proceed to feature extraction.")
    else:
        print("Issues found above — resolve before starting feature extraction.")


if __name__ == "__main__":
    main()