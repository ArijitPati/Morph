import soundfile as sf
import pandas as pd
from pathlib import Path

rows = []
for source in ["deep_voice", "gary_stafford"]:
    for label in ["real", "fake"]:
        folder = Path(f"data/raw/{source}/{label}")
        for f in folder.glob("*"):
            try:
                info = sf.info(str(f))
                rows.append({
                    "filepath": str(f),
                    "source": source,
                    "label": label,
                    "duration_sec": info.frames / info.samplerate,
                    "sample_rate": info.samplerate,
                    "channels": info.channels,
                    "corrupt": False,
                })
            except Exception as e:
                rows.append({"filepath": str(f), "source": source, "label": label,
                              "duration_sec": None, "sample_rate": None,
                              "channels": None, "corrupt": True})

df = pd.DataFrame(rows)
df.to_csv("data/metadata/audit_report.csv", index=False)
print(df.groupby(["source", "label"]).agg(
    count=("filepath", "count"),
    corrupt=("corrupt", "sum"),
    mean_duration=("duration_sec", "mean"),
    sample_rates=("sample_rate", lambda x: sorted(x.dropna().unique()))
))