from pathlib import Path
import pandas as pd

datasets = {
    "FS369": Path(r"F:\FinalSpark\fs369data"),
    "FS437": Path(r"F:\FinalSpark\fs437\fs437_export"),
}

fields = [
    "Batch Start",
    "Date",
    "Dead Date",
    "Lifespan",
    "Dead Cause",
    "Number of NS",
    "setup",
]

for name, folder in datasets.items():
    path = folder / f"{name.lower()}_package.hdf5"

    with pd.HDFStore(str(path), mode="r") as store:
        metadata = store.get(
            f"/{name.lower()}_wholelife_metadata"
        )

    print(f"\n===== {name} =====")

    for field in fields:
        value = metadata.iloc[0].get(field, "MISSING")
        print(f"{field}: {value}")

print("\nLocal metadata review complete.")