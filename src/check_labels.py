from pathlib import Path
import pandas as pd

datasets = {
    "FS437": Path(r"F:\FinalSpark\fs437\fs437_export"),
    "FS369": Path(r"F:\FinalSpark\fs369data"),
}

fields = [
    "Batch Name",
    "Batch Start",
    "Date",
    "Dead Date",
    "Lifespan",
    "Dead Cause",
    "MEA_date",
    "Number of NS",
    "setup",
]

def is_populated(value):
    if value is None:
        return False
    if isinstance(value, str):
        return value.strip().lower() not in (
            "", "nan", "nat", "none", "null", "n/a"
        )
    if isinstance(value, (list, tuple, dict)):
        return len(value) > 0
    try:
        return bool(pd.notna(value))
    except (TypeError, ValueError):
        return True

for name, folder in datasets.items():
    file = folder / f"{name.lower()}_package.hdf5"

    with pd.HDFStore(str(file), mode="r") as store:
        metadata = store.get(f"/{name.lower()}_wholelife_metadata")

    print(f"\n=== {name} METADATA ===")
    print("Metadata rows:", len(metadata))

    for field in fields:
        if field not in metadata.columns:
            print(f"{field}: MISSING COLUMN")
            continue

        values = metadata[field].tolist()
        count = sum(is_populated(value) for value in values)
        types = sorted({type(value).__name__ for value in values})

        print(
            f"{field}: {count}/{len(values)} populated "
            f"| types: {', '.join(types)}"
        )

print("\nDONE — No raw recordings opened.")
