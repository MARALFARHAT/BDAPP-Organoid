from pathlib import Path
from collections import Counter

import pandas as pd
import matplotlib.pyplot as plt

ROOT = Path(r"F:\FinalSpark\fs437\fs437_export")
OUT = Path(r"F:\BDAPP-Organoid\outputs")
OUT.mkdir(parents=True, exist_ok=True)

counts = Counter()
electrodes = set()
total = 0

print("Processing real FS437 events...")

with pd.HDFStore(
    str(ROOT / "fs437_package.hdf5"), mode="r"
) as store:

    chunks = store.select(
        "/fs437_wholelife_events",
        columns=["time_of_event", "electrode"],
        iterator=True,
        chunksize=100_000
    )

    for chunk in chunks:
        total += len(chunk)

        electrodes.update(
            chunk["electrode"].dropna().unique().tolist()
        )

        bins = (
            chunk["time_of_event"]
            .dropna()
            .dt.floor("6h")
            .value_counts()
        )

        counts.update(bins.to_dict())

        print(f"Processed {total:,} events")

timeline = pd.Series(
    dict(counts), name="detected_events"
).sort_index()

timeline.index.name = "window_start"

csv_path = OUT / "fs437_6hour_event_counts.csv"
timeline.to_csv(csv_path)

print("\nTotal events:", f"{total:,}")
print("Unique electrodes:", len(electrodes))
print("Six-hour windows with events:", len(timeline))

print("\nChecking segment index...")

index = pd.read_parquet(
    ROOT / "fs437_segment_index.parquet"
)

print("Index rows:", len(index))
print("Index columns:", index.columns.tolist())
print("Index data types:")
print(index.dtypes.to_string())

if {"t_start", "t_end"}.issubset(index.columns):
    starts = index["t_start"]
    ends = index["t_end"]

    if (
        pd.api.types.is_datetime64_any_dtype(starts)
        and pd.api.types.is_datetime64_any_dtype(ends)
    ):
        valid = (ends >= starts) & starts.notna() & ends.notna()
        print("Valid segment time ranges:", int(valid.sum()))
        print("Invalid or missing ranges:", int((~valid).sum()))
    else:
        print("Segment time units require further verification.")

fig, ax = plt.subplots(figsize=(12, 5))

ax.scatter(
    timeline.index,
    timeline.values,
    s=14,
    alpha=0.7
)

ax.set_title(
    "FS437 — Detected Neural Events Over Time"
)
ax.set_xlabel("Recorded timestamp")
ax.set_ylabel("Detected events per 6-hour clock window")
ax.grid(True, alpha=0.25)

fig.autofmt_xdate()
fig.tight_layout()

plot_path = OUT / "fs437_6hour_event_timeline.png"
fig.savefig(plot_path, dpi=180)
plt.close(fig)

print("\nSaved:", csv_path)
print("Saved:", plot_path)

print("\nIMPORTANT:")
print("Counts are not normalized for recording exposure.")
print("Missing time windows are not assumed to be zero activity.")
print("No raw waveform recordings were opened.")
print("\nANALYSIS COMPLETE")
