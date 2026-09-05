import os
import csv
from collections import Counter
from pathlib import Path

# Update this path if your extracted root directory is different
ASVSPOOF_ROOT = Path(r"C:\Users\Akasdip\morph\data\raw\asvspoof2019_la")
OUTPUT_CSV = Path(r"C:\Users\Akasdip\morph\data\metadata\asvspoof2019_la_manifest.csv")

# Map subsets to target audio subfolders and candidate protocol filenames
SUBSETS = {
    "train": ("ASVspoof2019_LA_train", ["ASVspoof2019.LA.cm.train.trn", "ASVspoof2019.LA.cm.train.trn.txt"]),
    "dev":   ("ASVspoof2019_LA_dev",   ["ASVspoof2019.LA.cm.dev.trl", "ASVspoof2019.LA.cm.dev.trl.txt"]),
    "eval":  ("ASVspoof2019_LA_eval",  ["ASVspoof2019.LA.cm.eval.trl", "ASVspoof2019.LA.cm.eval.trl.txt"]),
}

# Resolve nested directory path if present
if (ASVSPOOF_ROOT / "LA").exists():
    BASE_DIR = ASVSPOOF_ROOT / "LA"
else:
    BASE_DIR = ASVSPOOF_ROOT

PROTOCOL_DIR = BASE_DIR / "ASVspoof2019_LA_cm_protocols"

rows = []

for subset, (audio_dirname, candidate_protocols) in SUBSETS.items():
    protocol_path = None
    for cand in candidate_protocols:
        p = PROTOCOL_DIR / cand
        if p.exists():
            protocol_path = p
            break

    audio_dir = BASE_DIR / audio_dirname / "flac"

    if not protocol_path:
        print(f"[skip] protocol file not found for {subset} in: {PROTOCOL_DIR}")
        continue
    if not audio_dir.exists():
        print(f"[skip] audio dir not found for {subset}: {audio_dir}")
        continue

    print(f"Processing {subset} using protocol: {protocol_path.name}")
    with open(protocol_path, "r", encoding="utf-8") as f:
        for line in f:
            parts = line.strip().split()
            if len(parts) < 5:
                continue

            speaker_id, utt_id, _, system_id, key = parts[:5]
            label = "bonafide" if key == "bonafide" else "spoof"
            audio_path = audio_dir / f"{utt_id}.flac"

            if not audio_path.exists():
                print(f"[warn] missing audio file: {audio_path}")
                continue

            rows.append({
                "source": "asvspoof2019_la",
                "subset": subset,
                "speaker_id": speaker_id,
                "utterance_id": utt_id,
                "system_id": system_id,
                "label": label,
                "audio_path": str(audio_path.resolve()),
            })

if not rows:
    print("\n[Error] No records parsed. Please verify the folder contents of ASVSPOOF_ROOT.")
else:
    OUTPUT_CSV.parent.mkdir(parents=True, exist_ok=True)
    with open(OUTPUT_CSV, "w", newline="", encoding="utf-8") as f:
        writer = csv.DictWriter(f, fieldnames=rows[0].keys())
        writer.writeheader()
        writer.writerows(rows)

    print(f"\nSuccessfully wrote {len(rows)} rows to {OUTPUT_CSV}")

    counts = Counter((r["subset"], r["label"]) for r in rows)
    for k, v in sorted(counts.items()):
        print(f"Subset: {k[0]:<5} | Label: {k[1]:<8} | Count: {v}")