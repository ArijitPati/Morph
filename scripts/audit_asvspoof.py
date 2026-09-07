"""
audit_asvspoof.py
Reads the ASVspoof 2019 LA manifest CSV and checks each referenced
audio file for corruption, duration, and sample rate.
"""

import csv
import soundfile as sf
from collections import defaultdict

MANIFEST_CSV = r"C:\Users\Akasdip\morph\data\metadata\asvspoof2019_la_manifest.csv"

# groups[(subset, label)] -> list of (duration, sample_rate)
groups = defaultdict(list)
corrupt_counts = defaultdict(int)

with open(MANIFEST_CSV, "r") as f:
    reader = csv.DictReader(f)
    rows = list(reader)

print(f"Auditing {len(rows)} files...")

for i, row in enumerate(rows):
    key = (row["subset"], row["label"])
    path = row["audio_path"]
    try:
        info = sf.info(path)
        duration = info.frames / info.samplerate
        groups[key].append((duration, info.samplerate))
    except Exception as e:
        corrupt_counts[key] += 1
        print(f"[corrupt] {path}: {e}")

    if (i + 1) % 2000 == 0:
        print(f"  ...{i+1}/{len(rows)} checked")

print("\n--- SUMMARY ---")
print(f"{'subset':<8} {'label':<10} {'count':<7} {'corrupt':<8} {'mean_duration':<15} {'sample_rates'}")
for key in sorted(groups.keys()):
    subset, label = key
    durations = [d for d, sr in groups[key]]
    sample_rates = sorted(set(sr for d, sr in groups[key]))
    mean_dur = sum(durations) / len(durations) if durations else 0
    n_corrupt = corrupt_counts[key]
    print(f"{subset:<8} {label:<10} {len(durations):<7} {n_corrupt:<8} {mean_dur:<15.3f} {sample_rates}")