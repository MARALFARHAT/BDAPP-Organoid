from pathlib import Path
from collections import Counter
import pandas as pd

root = Path(r"F:\FinalSpark\fs437\fs437_export")
out = Path(r"F:\BDAPP-Organoid\outputs")

events = pd.read_csv(
    out / "fs437_6hour_event_counts.csv",
    parse_dates=["window_start"]
).set_index("window_start")

stim = pd.read_csv(
    out / "fs437_events_vs_stimulation.csv",
    parse_dates=["window_start"]
).set_index("window_start")[["stimulation_records"]]

merged = events.join(stim, how="outer")
merged.index = pd.to_datetime(merged.index, utc=True)

tables = {
    "temperature_mean": "/fs437_wholelife_incubator_temperature",
    "co2_mean": "/fs437_wholelife_incubator_CO2",
    "o2_mean": "/fs437_wholelife_incubator_O2",
}

with pd.HDFStore(str(root / "fs437_package.hdf5"), mode="r") as store:
    for new_col, key in tables.items():
        df = store.select(key)

        df["time"] = pd.to_datetime(df["time"], errors="coerce", utc=True)

        value_cols = [c for c in df.columns if c not in ["time", "incubator_id"]]
        value_col = value_cols[0]

        df[value_col] = pd.to_numeric(df[value_col], errors="coerce")
        df = df.dropna(subset=["time", value_col])

        df["window_start"] = df["time"].dt.floor("6h")
        agg = df.groupby("window_start")[value_col].mean().rename(new_col)

        merged = merged.join(agg, how="outer")

merged = merged.sort_index()
merged.to_csv(out / "fs437_6hour_merged_features.csv")

print("MERGED TABLE CREATED")
print("=" * 50)
print("Rows:", len(merged))
print("Columns:", merged.columns.tolist())
print()
print(merged.head(10).to_string())
print()
print("Saved:", out / "fs437_6hour_merged_features.csv")
print("No raw waveform files were opened.")

