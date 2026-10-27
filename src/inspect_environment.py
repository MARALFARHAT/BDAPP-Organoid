from pathlib import Path
import pandas as pd
import matplotlib.pyplot as plt

root = Path(r"F:\FinalSpark\fs437\fs437_export")
output = Path(r"F:\BDAPP-Organoid\outputs")
output.mkdir(exist_ok=True)

tables = {
    "Temperature": "/fs437_wholelife_incubator_temperature",
    "CO2": "/fs437_wholelife_incubator_CO2",
    "O2": "/fs437_wholelife_incubator_O2",
}

fig, axes = plt.subplots(3, 1, figsize=(13, 10), sharex=True)

with pd.HDFStore(
    str(root / "fs437_package.hdf5"), mode="r"
) as store:

    for ax, (name, key) in zip(axes, tables.items()):
        print(f"\n=== {name} ===")

        if key not in store.keys():
            print("Table not found")
            continue

        df = store.select(key)

        print("Rows:", len(df))
        print("Columns:", df.columns.tolist())
        print("Types:")
        print(df.dtypes.to_string())

        if "time" not in df.columns:
            print("No time column found")
            continue

        df["time"] = pd.to_datetime(
            df["time"], errors="coerce", utc=True
        )

        candidates = [
            col for col in df.columns
            if col not in ("time", "incubator_id")
        ]

        numeric = {}
        for col in candidates:
            values = pd.to_numeric(df[col], errors="coerce")
            if values.notna().any():
                numeric[col] = values

        if not numeric:
            print("No numeric measurement column found")
            continue

        matching = [
            col for col in numeric
            if name.lower() in col.lower()
        ]
        value_col = matching[0] if matching else next(iter(numeric))

        df["measurement"] = numeric[value_col]
        df = df.dropna(subset=["time", "measurement"])

        print("Measurement column:", value_col)
        print("Valid observations:", len(df))
        print("First timestamp:", df["time"].min())
        print("Last timestamp:", df["time"].max())

        if "incubator_id" in df.columns:
            for identifier, group in df.groupby("incubator_id"):
                ax.scatter(
                    group["time"],
                    group["measurement"],
                    s=3,
                    alpha=0.4,
                    label=f"Incubator {identifier}"
                )
            if df["incubator_id"].nunique() > 1:
                ax.legend()
        else:
            ax.scatter(
                df["time"],
                df["measurement"],
                s=3,
                alpha=0.4
            )

        ax.set_title(f"FS437 — {name}")
        ax.set_ylabel(f"{name} (reported units)")
        ax.grid(alpha=0.2)

axes[-1].set_xlabel("Timestamp (UTC)")
fig.autofmt_xdate()
fig.tight_layout()

path = output / "fs437_incubator_overview.png"
fig.savefig(path, dpi=180)
plt.close(fig)

print("\nSaved:", path)
print("No raw waveform files were opened.")
