from pathlib import Path
import pandas as pd

root = Path(r"F:\FinalSpark\fs437\fs437_export")
out = Path(r"F:\BDAPP-Organoid\outputs")

df = pd.read_csv(
    out / "fs437_6hour_merged_features.csv",
    parse_dates=["window_start"]
)

print("=" * 55)
print("FS437 — MERGED DATA QUALITY CHECK")
print("=" * 55)

print("\nTotal windows:", len(df))
print("Unique timestamps:", df["window_start"].nunique())

print("\nAVAILABLE WINDOWS BY MEASUREMENT")
print(df.notna().sum().to_string())

print("\nMISSING VALUES")
print(df.isna().sum().to_string())

both = df[
    ["detected_events", "stimulation_records"]
].notna().all(axis=1)

print("\nWindows with both events and stimulation:", both.sum())

print("\nENVIRONMENTAL MEASUREMENT RANGES")
for col in ["temperature_mean", "co2_mean", "o2_mean"]:
    print(
        f"{col}: min={df[col].min():.4f}, "
        f"max={df[col].max():.4f}"
    )

print("\nINCUBATOR ID CHECK")

with pd.HDFStore(
    str(root / "fs437_package.hdf5"), mode="r"
) as store:

    for metric in ["temperature", "CO2", "O2"]:
        key = f"/fs437_wholelife_incubator_{metric}"
        data = store.select(key)

        counts = data["incubator_id"].value_counts(dropna=False)

        print(f"\n{metric}")
        print("Distinct incubators:", len(counts))
        print("Rows per incubator:", counts.tolist())

        if len(counts) > 1:
            print(
                "WARNING: Multiple incubators — "
                "review mapping before combining averages."
            )

print("\nQUALITY CHECK COMPLETE")
print("No raw waveform files opened.")
