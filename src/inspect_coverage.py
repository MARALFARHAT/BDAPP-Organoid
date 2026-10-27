import pandas as pd
from pathlib import Path

path = Path(
    r"F:\FinalSpark\fs437\fs437_export"
    r"\fs437_segment_index.parquet"
)

df = pd.read_parquet(path)

print("FS437 RECORDING SEGMENT INDEX")
print("=" * 55)

print("Index rows:", len(df))
print("Unique electrodes:", df["electrode"].nunique())

print("\nCOLUMN TYPES")
print(df.dtypes.to_string())

print("\nFIRST THREE SEGMENTS")
print(df.head(3).to_string(index=False))

print("\nTIME RANGE")
print("Earliest start:", df["t_start"].min())
print("Latest end:", df["t_end"].max())

print("\nROW RANGE VALIDATION")
valid = df["row_end"] > df["row_start"]

print("Valid row ranges:", int(valid.sum()))
print("Invalid row ranges:", int((~valid).sum()))

print("\nCHECK COMPLETE")
print("Raw waveform data was not opened.")
