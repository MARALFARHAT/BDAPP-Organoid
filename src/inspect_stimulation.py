from pathlib import Path
from collections import Counter
import pandas as pd
import matplotlib.pyplot as plt

root = Path(r"F:\FinalSpark\fs437\fs437_export")
out = Path(r"F:\BDAPP-Organoid\outputs")

counts = Counter()
total = 0

with pd.HDFStore(str(root / "fs437_package.hdf5"), mode="r") as store:
    chunks = store.select(
        "/fs437_wholelife_stimulations",
        columns=["time_of_stim"],
        iterator=True,
        chunksize=100_000
    )

    for chunk in chunks:
        dates = pd.to_datetime(
            chunk["time_of_stim"], errors="coerce"
        ).dropna()

        buckets = dates.dt.floor("6h").value_counts()
        counts.update(buckets.to_dict())
        total += len(chunk)

stim = pd.Series(
    counts, name="stimulation_records", dtype="int64"
).sort_index()
stim.index.name = "window_start"

events = pd.read_csv(
    out / "fs437_6hour_event_counts.csv",
    parse_dates=["window_start"]
).set_index("window_start")

comparison = events.join(stim, how="outer").sort_index()

comparison.to_csv(
    out / "fs437_events_vs_stimulation.csv"
)

print("FS437 STIMULATION ANALYSIS")
print("=" * 50)
print("Total stimulation records:", total)
print("Windows containing events:", events.shape[0])
print("Windows containing stimulation:", stim.shape[0])
print(
    "Windows containing both:",
    comparison[["detected_events", "stimulation_records"]]
    .notna().all(axis=1).sum()
)

print("\nFIRST WINDOWS")
print(comparison.head(10).to_string())

print("\nLAST WINDOWS")
print(comparison.tail(10).to_string())

fig, ax = plt.subplots(figsize=(12, 5))
ax.scatter(stim.index, stim.values, s=25)

ax.set_title("FS437 - Stimulation Records Over Time")
ax.set_xlabel("Recorded timestamp (UTC)")
ax.set_ylabel("Stimulation records per 6-hour window")
ax.grid(True, alpha=0.25)

fig.autofmt_xdate()
fig.tight_layout()
fig.savefig(
    out / "fs437_stimulation_timeline.png", dpi=180
)
plt.close(fig)

print("\nANALYSIS COMPLETE")
print("Raw waveform files were not opened.")
print("Counts do not establish recording exposure.")
