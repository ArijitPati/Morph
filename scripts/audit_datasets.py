import soundfile as sf
import pandas as pd
import csv
from pathlib import Path

rows = []

# --- Existing sources: deep_voice, gary_stafford ---
for source in ["deep_voice", "gary_stafford"]:
    for label in ["real", "fake"]:
        folder = Path(f"data/raw/{source}/{label}")
        for f in folder.glob("*"):
            try:
                info = sf.info(str(f))
                rows.append({
                    "filepath": str(f),
                    "source": source,
                    "subset": None,          # not applicable to these sources
                    "label": label,
                    "duration_sec": info.frames / info.samplerate,
                    "sample_rate": info.samplerate,
                    "channels": info.channels,
                    "corrupt": False,
                })
            except Exception as e:
                rows.append({"filepath": str(f), "source": source, "subset": None,
                              "label": label, "duration_sec": None, "sample_rate": None,
                              "channels": None, "corrupt": True})

# --- New source: asvspoof2019_la, read from its manifest instead of a folder scan ---
ASVSPOOF_MANIFEST = "data/metadata/asvspoof2019_la_manifest.csv"

with open(ASVSPOOF_MANIFEST, "r") as f:
    manifest_rows = list(csv.DictReader(f))

print(f"Auditing {len(manifest_rows)} ASVspoof files...")

for i, m in enumerate(manifest_rows):
    path = m["audio_path"]
    # normalize label to real/fake to match the other two sources' convention in this file
    label = "real" if m["label"] == "bonafide" else "fake"
    try:
        info = sf.info(path)
        rows.append({
            "filepath": path,
            "source": "asvspoof2019_la",
            "subset": m["subset"],           # train/dev/eval
            "label": label,
            "duration_sec": info.frames / info.samplerate,
            "sample_rate": info.samplerate,
            "channels": info.channels,
            "corrupt": False,
        })
    except Exception as e:
        rows.append({"filepath": path, "source": "asvspoof2019_la", "subset": m["subset"],
                      "label": label, "duration_sec": None, "sample_rate": None,
                      "channels": None, "corrupt": True})

    if (i + 1) % 5000 == 0:
        print(f"  ...{i+1}/{len(manifest_rows)} checked")

# --- Write unified per-file audit report ---
df = pd.DataFrame(rows)
df.to_csv("data/metadata/audit_report.csv", index=False)

print(df.groupby(["source", "subset", "label"], dropna=False).agg(
    count=("filepath", "count"),
    corrupt=("corrupt", "sum"),
    mean_duration=("duration_sec", "mean"),
    sample_rates=("sample_rate", lambda x: sorted(x.dropna().unique()))
))