import pandas as pd

path = r"F:\FinalSpark\fs437\fs437_export\fs437_package.hdf5"
key = "/fs437_wholelife_events"

with pd.HDFStore(path, mode="r") as store:
    total = store.get_storer(key).nrows

    print("FS437 TOTAL EVENT RECORDS:", total)

    windows = {
        "Beginning": 0,
        "Middle": max(0, total // 2 - 10000),
        "End": max(0, total - 20000)
    }

    for name, start in windows.items():
        events = store.select(
            key, start=start, stop=min(start + 20000, total)
        )

        times = events["time_of_event"]

        print(f"\n--- {name} ---")
        print("Sampled events:", len(events))
        print("Timestamp type:", times.dtype)
        print("Chronologically sorted:", times.is_monotonic_increasing)
        print("Active electrodes:", events["electrode"].nunique())

        if pd.api.types.is_datetime64_any_dtype(times):
            duration = (times.max() - times.min()).total_seconds()
            print("Sample duration (seconds):", round(duration, 2))
        else:
            print("Timestamp units need verification before calculating duration.")

print("\nTIMELINE INSPECTION COMPLETE")
