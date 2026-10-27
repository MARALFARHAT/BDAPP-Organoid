from pathlib import Path
import pandas as pd

path = Path(r"F:\FinalSpark\fs437\fs437_export\fs437_package.hdf5")

print("=" * 50)
print("BDAPP - FS437 DATASET INSPECTION")
print("=" * 50)

with pd.HDFStore(str(path), mode="r") as store:
    for key in store.keys():
        print(f"\nTABLE: {key}")

        storer = store.get_storer(key)

        if storer.is_table:
            data = store.select(key, start=0, stop=1)
        elif key.endswith("_metadata"):
            data = store.get(key)
        else:
            print("Skipped unsupported format")
            continue

        if hasattr(data, "dtypes"):
            for column, dtype in data.dtypes.items():
                print(f"  {column}: {dtype}")

print("\nINSPECTION COMPLETE")
