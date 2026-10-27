from pathlib import Path
import pandas as pd
import matplotlib.pyplot as plt

DATA = Path(
    r"F:\FinalSpark\fs437\fs437_export\fs437_package.hdf5"
)
OUTPUT = Path(r"F:\BDAPP-Organoid\outputs")
OUTPUT.mkdir(parents=True, exist_ok=True)

print("Reading a small sample of real FS437 events...")

with pd.HDFStore(str(DATA), mode="r") as store:
    events = store.select(
        "/fs437_wholelife_events",
        start=0,
        stop=20000
    )

print("Events loaded:", len(events))
print("Available columns:", events.columns.tolist())

counts = events["electrode"].value_counts().sort_index()

fig, ax = plt.subplots(figsize=(11, 5))
ax.bar(counts.index.astype(str), counts.values)

ax.set_title("FS437 - Detected Neural Events by Electrode")
ax.set_xlabel("Electrode")
ax.set_ylabel("Detected events in sampled rows")
ax.tick_params(axis="x", rotation=60)

plt.tight_layout()

save_path = OUTPUT / "fs437_electrode_activity_preview.png"
plt.savefig(save_path, dpi=180)
plt.close(fig)

print("Figure saved:", save_path)
print("Done! No raw waveform file was opened.")
