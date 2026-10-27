# Maral Farhatnejad | Student ID: 23012856 | BDAPP
from pathlib import Path
from collections import Counter
from datetime import datetime, timezone
import json
import sys
import numpy as np
import pandas as pd
import matplotlib
matplotlib.use("Agg")
import matplotlib.pyplot as plt


def read_chunks(store, key, columns):
    if not store.get_storer(key).is_table:
        raise ValueError(f"Expected a chunk-readable table: {key}")
    return store.select(
        key, columns=columns, iterator=True, chunksize=100000
    )


def utc(values):
    if pd.api.types.is_numeric_dtype(values.dtype):
        raise ValueError("Numeric timestamp units need verification.")
    return pd.to_datetime(values, errors="coerce", utc=True)


def run(source, output_root):
    if not source.is_file():
        raise FileNotFoundError(f"FS369 package not found: {source}")

    folder = output_root / datetime.now(timezone.utc).strftime(
        "%Y%m%dT%H%M%S%fZ"
    )
    folder.mkdir(parents=True, exist_ok=False)
    audit = {}

    with pd.HDFStore(str(source), mode="r") as store:
        keys = store.keys()
        frames = []

        for kind, table, time_col in [
            ("detected_events", "events", "time_of_event"),
            ("stimulation_records", "stimulations", "time_of_stim"),
        ]:
            key = f"/fs369_wholelife_{table}"

            if key not in keys:
                if table == "events":
                    raise KeyError(f"Required table absent: {key}")
                print(f"Missing table: {key}", flush=True)
                continue

            counts, channels = Counter(), Counter()
            rows = bad_times = 0
            columns = (
                [time_col, "electrode"]
                if table == "events"
                else [time_col]
            )

            for number, part in enumerate(
                read_chunks(store, key, columns), 1
            ):
                times = utc(part[time_col])
                valid = times.notna()

                counts.update(
                    times[valid].dt.floor("6h").value_counts().to_dict()
                )

                rows += len(part)
                bad_times += int((~valid).sum())

                if table == "events":
                    use = valid & part.electrode.notna()
                    tiny = pd.DataFrame({
                        "window_start": times[use].dt.floor("6h"),
                        "electrode": part.loc[
                            use, "electrode"
                        ].astype(str),
                    })
                    channels.update(
                        tiny.groupby(
                            ["window_start", "electrode"]
                        ).size().to_dict()
                    )

                if number == 1 or number % 10 == 0:
                    print(
                        f"{kind}: {rows:,} rows processed",
                        flush=True,
                    )

            frame = pd.Series(
                dict(counts), name=kind, dtype="int64"
            ).to_frame()

            frame.index = (
                pd.DatetimeIndex(frame.index)
                if len(frame)
                else pd.DatetimeIndex([], tz="UTC")
            )

            frames.append(frame)
            audit[kind] = {
                "source_rows": rows,
                "invalid_time_rows": bad_times,
            }

            if table == "events":
                if frame.empty:
                    raise ValueError(
                        "No events have usable timestamps."
                    )

                per_channel = pd.DataFrame(
                    [
                        (t, e, n)
                        for (t, e), n in channels.items()
                    ],
                    columns=[
                        "window_start",
                        "electrode",
                        "detected_events",
                    ],
                )

                per_channel.to_csv(
                    folder / "event_electrode_counts.csv",
                    index=False,
                )

                if len(per_channel):
                    frame["active_observed_electrodes"] = (
                        per_channel.groupby(
                            "window_start"
                        ).electrode.nunique()
                    )

        environment_rows = []

        for metric, output_col in [
            ("temperature", "temperature_mean"),
            ("CO2", "co2_mean"),
            ("O2", "o2_mean"),
        ]:
            key = f"/fs369_wholelife_incubator_{metric}"

            if key not in keys:
                continue

            value_col = f"incubator_{metric}"
            sums, nums = Counter(), Counter()
            source_rows = rejected = 0

            for part in read_chunks(
                store, key, ["time", "incubator_id", value_col]
            ):
                times = utc(part.time)
                values = pd.to_numeric(
                    part[value_col], errors="coerce"
                )

                valid = (
                    times.notna()
                    & values.notna()
                    & np.isfinite(values)
                )

                source_rows += len(part)
                rejected += int((~valid).sum())

                ids = (
                    part.incubator_id
                    .fillna("__MISSING__")
                    .astype(str)
                    .replace("", "__MISSING__")
                )

                tiny = pd.DataFrame({
                    "window_start": times[valid].dt.floor("6h"),
                    "incubator_id": ids[valid],
                    "value": values[valid],
                })

                grouped = tiny.groupby(
                    ["window_start", "incubator_id"]
                ).value

                sums.update(grouped.sum().to_dict())
                nums.update(grouped.count().to_dict())

            items = [
                {
                    "window_start": t,
                    "incubator_id": identifier,
                    "metric": metric,
                    "mean": sums[(t, identifier)] / n,
                    "observations": n,
                }
                for (t, identifier), n in nums.items()
            ]

            environment_rows.extend(items)
            identifiers = {
                item["incubator_id"] for item in items
            }

            audit[metric] = {
                "source_rows": source_rows,
                "invalid_time_or_value_rows": rejected,
                "distinct_incubators": len(identifiers),
            }

            if (
                len(identifiers) == 1
                and "__MISSING__" not in identifiers
            ):
                frames.append(
                    pd.Series(
                        {
                            item["window_start"]: item["mean"]
                            for item in items
                        },
                        name=output_col,
                    ).to_frame()
                )
            else:
                print(
                    f"{metric}: kept incubators separate; "
                    "see environment_by_incubator.csv"
                )

        pd.DataFrame(
            environment_rows,
            columns=[
                "window_start",
                "incubator_id",
                "metric",
                "mean",
                "observations",
            ],
        ).to_csv(
            folder / "environment_by_incubator.csv",
            index=False,
        )

        key = "/fs369_wholelife_metadata"

        if key in keys:
            storer = store.get_storer(key)
            meta = (
                store.select(key, start=0, stop=1000)
                if storer.is_table
                else store.get(key)
            )

            audit["metadata_rows_read"] = len(meta)

            (folder / "metadata.json").write_text(
                meta.to_json(
                    orient="records",
                    date_format="iso",
                    default_handler=str,
                ),
                encoding="utf-8",
            )

    merged = pd.concat(frames, axis=1).sort_index()
    merged = merged.reindex(
        pd.date_range(
            merged.index.min(),
            merged.index.max(),
            freq="6h",
        )
    )
    merged.index.name = "window_start"

    columns = [
        "detected_events",
        "stimulation_records",
        "temperature_mean",
        "co2_mean",
        "o2_mean",
    ]

    for col in columns:
        if col not in merged:
            merged[col] = np.nan

    merged.to_csv(folder / "merged_window_features.csv")

    quality = pd.DataFrame({
        "available_windows": merged.notna().sum(),
        "missing_windows": merged.isna().sum(),
    })
    quality.to_csv(folder / "missingness.csv")

    audit["models_trained"] = False
    audit["notes"] = [
        "Counts are not exposure-normalized firing rates.",
        "Missing windows are not confirmed silence.",
        "Incubator means retain source units and are sample-weighted.",
        "Observed electrodes are not independent organoids.",
        "No deduplication or stimulation-artifact exclusion applied.",
        "No biological age or death labels inferred.",
    ]

    (folder / "analysis_summary.json").write_text(
        json.dumps(audit, indent=2),
        encoding="utf-8",
    )

    fig, axes = plt.subplots(
        5, 1,
        figsize=(13, 12),
        sharex=True,
        constrained_layout=True,
    )

    for ax, col in zip(axes, columns):
        ax.plot(
            merged.index,
            merged[col],
            "o-",
            markersize=3,
        )
        ax.set_title(col, loc="left")
        ax.set_ylabel(
            "Records" if col in columns[:2] else "Reported value"
        )
        ax.grid(alpha=.2)

    axes[-1].set_xlabel(
        "Six-hour window start (UTC). Missing values remain gaps."
    )
    fig.suptitle(
        "FS369: recorded activity, stimulation and environment"
    )
    fig.savefig(
        folder / "combined_overview.png",
        dpi=160,
    )
    plt.close(fig)

    print("\nFS369 COMPLETE")
    print(quality.to_string())
    print("\nSaved:", folder)
    print(
        "No raw waveforms opened. "
        "No model trained in this step."
    )

    return merged, audit


if __name__ == "__main__":
    source = (
        Path(sys.argv[1])
        if len(sys.argv) > 1
        else Path(
            r"F:\FinalSpark\fs369data\fs369_package.hdf5"
        )
    )

    out = (
        Path(sys.argv[2])
        if len(sys.argv) > 2
        else Path(r"F:\BDAPP-Organoid\outputs\FS369")
    )

    run(source, out)
