from pathlib import Path
import pandas as pd
import matplotlib.pyplot as plt
import matplotlib.dates as mdates

root = Path(r"F:\BDAPP-Organoid")
source = root / "outputs/fs437_6hour_merged_features.csv"

df = pd.read_csv(source)

df["window_start"] = pd.to_datetime(
    df["window_start"], utc=True
)

df = df.set_index("window_start").sort_index()

columns = [
    ("detected_events", "Detected neural events", "Events"),
    ("stimulation_records", "Electrical stimulation", "Records"),
    ("temperature_mean", "Incubator temperature", "Reported value"),
    ("co2_mean", "Incubator CO2", "Reported value"),
    ("o2_mean", "Incubator O2", "Reported value")
]

fig, axes = plt.subplots(
    5, 1,
    figsize=(14, 13),
    sharex=True,
    constrained_layout=True
)

for i, (column, title, ylabel) in enumerate(columns):

    ax = axes[i]
    series = df[column]

    if i < 2:
        valid = series.dropna()

        ax.bar(
            valid.index,
            valid.values,
            width=0.20,
            alpha=0.8
        )

        missing = series[series.isna()].index

        for timestamp in missing:
            ax.axvspan(
                timestamp,
                timestamp + pd.Timedelta(hours=6),
                color="gray",
                alpha=0.15
            )

    else:
        ax.plot(
            series.index,
            series.values,
            marker="o",
            markersize=3,
            linewidth=1.3
        )

    ax.set_title(title, loc="left", fontsize=11)
    ax.set_ylabel(ylabel)
    ax.grid(alpha=0.2)

axes[-1].xaxis.set_major_formatter(
    mdates.DateFormatter("%b %d", tz=mdates.UTC)
)

axes[-1].set_xlabel("Recorded date (UTC)")

fig.suptitle(
    "FS437: Neural Activity, Stimulation and Environment",
    fontsize=15,
    fontweight="bold"
)

path = root / "outputs/fs437_combined_overview.png"

fig.savefig(path, dpi=200)
plt.close(fig)

print("COMBINED VISUALIZATION CREATED")
print("Saved:", path)
print("Gray shading indicates missing records, not zero activity.")
print("Event counts are not exposure-normalized.")
print("No raw waveform data was opened.")
