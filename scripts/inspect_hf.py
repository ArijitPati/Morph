from datasets import load_dataset, Audio

ds = load_dataset("garystafford/deepfake-audio-detection")
ds = ds.cast_column("audio", Audio(decode=False))   # stops it from trying to decode via torchcodec

print(ds)
print(ds["train"].features)

row = ds["train"][0]
print(row["label"])
print(row["audio"].keys())      # -> dict_keys(['bytes', 'path'])