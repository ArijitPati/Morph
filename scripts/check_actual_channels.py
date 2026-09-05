"""
check_actual_channels.py

Checks the actual channel count of every file referenced in
master_metadata.csv (the files that will really be used for feature
extraction) — grouped by source AND label, since audit_report.csv's
source-only grouping masked per-label differences.
"""

import csv
import soundfile as sf
from pathlib import Path
from collections import defaultdict

MASTER_METADATA_CSV = Path("data/metadata/master_metadata.csv")

def main():
    with open(MASTER_METADATA_CSV, "r") as f:
        rows = list(csv.DictReader(f))

    print(f"Checking channels for {len(rows)} files referenced in master_metadata.csv...")

    channel_counts = defaultdict(lambda: defaultdict(int))  # (source,label) -> channels -> count
    failed = []

    for i, row in enumerate(rows):
        path = row["audio_path"]
        key = (row["source"], row["label"])
        try:
            info = sf.info(path)
            channel_counts[key][info.channels] += 1
        except Exception as e:
            failed.append((path, str(e)))

        if (i + 1) % 20000 == 0:
            print(f"  ...{i+1}/{len(rows)} checked")

    print("\n=== Channel breakdown by source + label (actual files used) ===")
    problems = False
    for key in sorted(channel_counts.keys()):
        source, label = key
        counts = dict(channel_counts[key])
        if list(counts.keys()) != [1]:
            problems = True
            print(f"[WARNING] {source} / {label}: {counts}")
        else:
            print(f"[ok] {source} / {label}: {counts}")

    if failed:
        print(f"\n{len(failed)} files failed to read (see above logic if needed).")

    print("\n=== VERDICT ===")
    if not problems:
        print("All referenced files are mono. Safe to proceed to feature extraction.")
    else:
        print("Some referenced files are still stereo — these need mono conversion before extraction.")

if __name__ == "__main__":
    main()