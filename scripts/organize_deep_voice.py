import shutil
from pathlib import Path

SRC = Path("data/raw/deep_voice")
DST_REAL = SRC / "real"
DST_FAKE = SRC / "fake"
DST_REAL.mkdir(exist_ok=True)
DST_FAKE.mkdir(exist_ok=True)

# Adjust these two paths after checking the actual unzip output above
for f in SRC.rglob("*"):
    if f.is_file() and f.suffix.lower() in (".wav", ".mp3"):
        if "REAL" in str(f).upper() and "DEMONSTRATION" not in str(f).upper():
            shutil.copy2(f, DST_REAL / f.name)
        elif "FAKE" in str(f).upper() and "DEMONSTRATION" not in str(f).upper():
            shutil.copy2(f, DST_FAKE / f.name)

print("REAL:", len(list(DST_REAL.glob("*"))))
print("FAKE:", len(list(DST_FAKE.glob("*"))))