# morph — Deepfake Audio Detection: Data Pipeline

## What this is

`morph` is a deepfake/spoofed-speech detection project. This document covers the **data ingestion, integrity, and preprocessing pipeline** — the stage that takes three raw, heterogeneous audio datasets and turns them into one unified, validated, split-ready dataset (`master_metadata.csv`). Feature extraction and model training are the next stage, not yet started.

## Datasets used

| Dataset | Source | Role |
|---|---|---|
| **DEEP-VOICE** | Kaggle `birdy654/deep-voice-deepfake-voice-recognition` | RVC voice-conversion impersonations of a small number of public figures. Long-form clips (~468s avg), only 64 files total. |
| **gary_stafford** | Hugging Face `garystafford/deepfake-audio-detection` | 1,866 short clips (~3–5s), balanced 933/933 real/fake, across 6 TTS platforms. |
| **ASVspoof 2019 LA** | Official DataShare / Zenodo | Large-scale (~130k files across train/dev/eval), academic-standard spoofing benchmark with official speaker-disjoint splits and attack-type labels. |

## Why three datasets, and why this order

DEEP-VOICE and gary_stafford were the original two-dataset plan. ASVspoof 2019 LA was added specifically because the other two, on their own, could not support a genuine speaker-disjoint train/test split — DEEP-VOICE's speaker pool is far too narrow (a couple of public figures), and gary_stafford has no speaker metadata at all. ASVspoof's official, expert-curated train/dev/eval structure fixes this gap and became the backbone of the final split strategy.

## Pipeline stages (in order run)

1. **Project scaffolding** — `data/raw/`, `data/interim/`, `data/metadata/`, `scripts/`
2. **Environment setup** — Python venv + `requirements.txt`
3. **Dataset ingestion**
   - `organize_deep_voice.py` — Kaggle download → flattened into `data/raw/deep_voice/{real,fake}/`
   - `export_gary_stafford.py` — Hugging Face `datasets` → WAV export into `data/raw/gary_stafford/{real,fake}/` (uses `Audio(decode=False)` + manual `soundfile` decoding to avoid a `torchcodec` dependency)
   - `organize_asvspoof.py` — parses ASVspoof's protocol files into `asvspoof2019_la_manifest.csv`, **referencing original file paths rather than copying audio** (ASVspoof is large; copying would double storage for no benefit)
4. **Integrity audit** — `audit_datasets.py` checks every file (via folder scan for deep_voice/gary_stafford, via manifest for ASVspoof) for corruption, sample rate, channels, duration. Result: **0 corrupt files across all three sources.** Output: `data/metadata/audit_report.csv`.
5. **Standardization** — the audit surfaced two real problems, both fixed before merging:
   - **Sample-rate/label confound in gary_stafford**: all `real` clips were 44100Hz, all `fake` clips were 16000Hz — label was perfectly predictable from sample rate alone, a classic false-positive-cause (model/data mismatch) that would have produced misleadingly high, non-generalizing accuracy. Fixed via `resample_gary_stafford_real.py` (real class only; fake class was already 16000Hz).
   - **DEEP-VOICE format mismatch**: long-form (~468s) clips vs. everything else's ~3–5s scale, plus mixed sample rates (40k/44.1k/48k). Fixed via `chunk_deep_voice.py`: resample → RMS-energy VAD trim (removes leading/trailing silence) → split into 4-second, non-overlapping windows → discard windows still mostly silent, discard trailing remainders under 2s. Produced 7,237 chunks (836 real from 8 files, 6,401 fake from 56 files) with a `parent_file`/`chunk_index` manifest for traceability.
6. **Unified metadata build** — `build_metadata.py` merges all three sources into `data/metadata/master_metadata.csv` with a consistent schema (`audio_path`, `source`, `label`, `split`, `speaker_id`, `system_id`, `parent_file`, `chunk_index`, `sample_rate`, `duration_sec`), applying the split strategy below.
7. **Final validation** — `validate_metadata.py` (path existence + ASVspoof duration backfill) and `check_actual_channels.py` (channel consistency, checked against the *actual* files referenced in `master_metadata.csv`, not just the raw audit) confirm the pipeline output is clean: all 130,564 referenced files resolve, and all are confirmed mono.

## Labeling convention

`bonafide` / `spoof` — not `real`/`fake` — chosen from the start for direct compatibility with ASVspoof's own convention and the wider anti-spoofing literature.

## Split strategy

| Split | ASVspoof 2019 LA | gary_stafford | DEEP-VOICE |
|---|---|---|---|
| **train** | official `train` subset | ~70%, stratified by label | — |
| **val** | official `dev` subset | ~15%, stratified by label | — |
| **test** | official `eval` subset | ~15%, stratified by label | **100%** |

**Why this split, not a naive pooled random split:**
- ASVspoof's official subsets are speaker-disjoint by design (and `eval` includes attack types unseen in `train`) — re-splitting randomly would risk speaker leakage, a documented failure mode in anti-spoofing research that inflates apparent accuracy.
- gary_stafford has no speaker or platform metadata, so a true speaker-disjoint split isn't possible for it. It's split by stratified random sampling instead — **this is a known, documented limitation**: any performance attributable to gary_stafford alone should be read as an upper-bound estimate, not a guaranteed generalization result.
- DEEP-VOICE's speaker pool (2 people) is too narrow to split meaningfully at all. It's used entirely as a **held-out generalization probe** — a model trained without ever seeing these speakers or this voice-conversion method should still be evaluated on how well it detects them.

## Final dataset composition

| Split | bonafide | spoof | ratio |
|---|---|---|---|
| train | 3,233 | 23,453 | ~7.3 : 1 |
| val | 2,688 | 22,436 | ~8.3 : 1 |
| test | 8,331 | 70,423 | ~8.5 : 1 |

**Total: 130,564 rows.** Class imbalance is a structural property of ASVspoof (many attack systems, one bonafide class) and must be handled at training time (`scale_pos_weight`, class-weighted loss, or stratified k-fold) — it is not a data quality issue.

## Important evaluation note for DEEP-VOICE

DEEP-VOICE's 7,237 test chunks come from only 64 source files (8 real, 56 fake) — chunks from the same file are correlated, not independent samples. **Report both chunk-level and file-level (8 real / 56 fake) metrics**, and use per-class metrics (recall on bonafide, recall on spoof, or balanced accuracy) rather than pooled accuracy, since the 836/6,401 imbalance makes pooled accuracy misleading on its own.

## Directory structure

```
morph/
├── data/
│   ├── raw/                          # NOT in git — regenerate via scripts below
│   │   ├── deep_voice/{real,fake}/
│   │   └── gary_stafford/{real,fake}/
│   ├── interim/                      # NOT in git — regenerate via scripts below
│   │   ├── deep_voice_chunks/{real,fake}/
│   │   └── gary_stafford_real_16k/
│   └── metadata/                     # tracked in git — small CSVs
│       ├── audit_report.csv
│       ├── asvspoof2019_la_manifest.csv
│       ├── deep_voice_chunks_manifest.csv
│       ├── gary_stafford_real_16k_manifest.csv
│       └── master_metadata.csv
└── scripts/
    ├── organize_deep_voice.py
    ├── export_gary_stafford.py
    ├── organize_asvspoof.py
    ├── audit_datasets.py
    ├── resample_gary_stafford_real.py
    ├── chunk_deep_voice.py
    ├── build_metadata.py
    └── validate_metadata.py
```

## How to reproduce this pipeline from scratch

1. Download DEEP-VOICE from Kaggle, gary_stafford from Hugging Face, and ASVspoof 2019 LA (**LA only, not PA** — PA is replay-attack data, irrelevant to voice cloning) from the official source.
2. Run the scripts in the order listed in "Pipeline stages" above.
3. Run `validate_metadata.py` and confirm a clean verdict (no missing paths, all channels mono) before proceeding to feature extraction.

**Disk note:** keep raw archives only until extraction is verified, then delete them — the full pipeline doesn't need the original zips/archives once data is extracted and organized.

## Status

**Complete:** full data pipeline (ingestion → audit → standardization → unified metadata → validation) for all three sources.

**Not started:** feature extraction (planned: Librosa MFCC + delta-delta), model training, amplitude/loudness normalization (deliberately deferred to the feature-extraction stage).