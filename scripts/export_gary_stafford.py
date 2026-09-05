import io
import soundfile as sf
from pathlib import Path
from datasets import load_dataset, Audio

OUT = Path("data/raw/gary_stafford")
(OUT / "real").mkdir(parents=True, exist_ok=True)
(OUT / "fake").mkdir(parents=True, exist_ok=True)

ds = load_dataset("garystafford/deepfake-audio-detection")["train"]
ds = ds.cast_column("audio", Audio(decode=False))
label_names = ds.features["label"].names  # ['real', 'fake']

failed = []
for i, row in enumerate(ds):
    label = label_names[row["label"]]
    folder = "real" if label == "real" else "fake"
    raw_bytes = row["audio"]["bytes"]

    try:
        data, sr = sf.read(io.BytesIO(raw_bytes))
        sf.write(OUT / folder / f"gs_{i:05d}.wav", data, sr)
    except Exception as e:
        failed.append((i, str(e)))

print(f"Exported {ds.num_rows - len(failed)} files, {len(failed)} failed.")
if failed:
    print(failed[:10])