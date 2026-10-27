# Maral Farhatnejad | Student ID: 23012856 | BDAPP
r"""Process FinalSpark event exports locally, without opening raw waveform files.

Example:
  python .\src\run_finalspark.py --dataset FS369
  python .\src\run_finalspark.py --dataset FS437

Outputs describe recorded events. Recording exposure, biological age, and
death endpoints are not inferred from the first or last recorded event.
"""
from __future__ import annotations

import argparse
from collections import Counter
from datetime import datetime, timezone
import json
from pathlib import Path
import platform
import sys

import numpy as np
import pandas as pd
import matplotlib
matplotlib.use("Agg")
import matplotlib.pyplot as plt
import matplotlib.dates as mdates

DEFAULT_INPUTS = {
    "FS369": Path(r"F:\FinalSpark\fs369data\fs369_package.hdf5"),
    "FS437": Path(r"F:\FinalSpark\fs437\fs437_export\fs437_package.hdf5"),
}
DEFAULT_OUTPUT = Path(r"F:\BDAPP-Organoid\outputs")
ENVIRONMENT = {
    "temperature": ("incubator_temperature", "temperature_mean"),
    "CO2": ("incubator_CO2", "co2_mean"),
    "O2": ("incubator_O2", "o2_mean"),
}


def log(message):
    print(message, flush=True)


def timestamps(values):
    # Numeric epochs require explicit units. Never assume nanoseconds.
    if pd.api.types.is_numeric_dtype(values.dtype):
        raise ValueError("Numeric timestamps found. Units must be verified first.")
    return pd.to_datetime(values, errors="coerce", utc=True)


def chunks(store, key, columns, size):
    if key not in store.keys():
        raise KeyError(f"Required table is absent: {key}")
    storer = store.get_storer(key)
    if not storer.is_table:
        raise ValueError(f"{key} is not a chunk-readable pandas table.")
    sample = store.select(key, start=0, stop=1)
    absent = set(columns) - set(sample.columns)
    if absent:
        raise ValueError(f"{key}: missing columns {sorted(absent)}; found {list(sample.columns)}")
    return store.select(key, columns=columns, iterator=True, chunksize=size)


def count_records(store, key, time_column, size, frequency, electrodes=False):
    counts, channel_counts = Counter(), Counter()
    audit = {"source_rows": 0, "valid_timestamp_rows": 0,
             "invalid_timestamp_rows": 0, "missing_electrode_valid_time_rows": 0}
    columns = [time_column, "electrode"] if electrodes else [time_column]
    for number, frame in enumerate(chunks(store, key, columns, size), 1):
        time = timestamps(frame[time_column])
        valid = time.notna()
        audit["source_rows"] += len(frame)
        audit["valid_timestamp_rows"] += int(valid.sum())
        audit["invalid_timestamp_rows"] += int((~valid).sum())
        windows = time.loc[valid].dt.floor(frequency)
        counts.update(windows.value_counts().to_dict())
        if electrodes:
            identified = valid & frame["electrode"].notna()
            audit["missing_electrode_valid_time_rows"] += int((valid & ~identified).sum())
            small = pd.DataFrame({
                "window_start": time.loc[identified].dt.floor(frequency),
                "electrode": frame.loc[identified, "electrode"].astype(str),
            })
            channel_counts.update(small.groupby(["window_start", "electrode"]).size().to_dict())
        if number == 1 or number % 10 == 0:
            log(f"  {key}: {audit['source_rows']:,} rows processed")
    series = pd.Series(dict(counts), dtype="int64").sort_index()
    series.index = pd.DatetimeIndex(series.index, tz="UTC") if not len(series) else pd.DatetimeIndex(series.index)
    series.index.name = "window_start"
    return series, channel_counts, audit


def process(store, dataset, out, chunk_rows=100_000, hours=6):
    prefix, freq = dataset.lower(), f"{hours}h"
    warnings, audit, metadata = [], {}, {}
    schema = []
    for key in store.keys():
        storer = store.get_storer(key)
        schema.append({"key": key, "table_format": bool(storer.is_table),
                       "rows": getattr(storer, "nrows", None)})
    pd.DataFrame(schema).to_csv(out / "schema.csv", index=False)

    key = f"/{prefix}_wholelife_metadata"
    if key in store.keys():
        storer = store.get_storer(key)
        if storer.is_table:
            frame = store.select(key, start=0, stop=1000)
            metadata["truncated"] = bool(storer.nrows and storer.nrows > 1000)
        else:
            frame = store.get(key)
        if isinstance(frame, pd.Series):
            frame = frame.to_frame().T
        metadata["records"] = frame.to_dict(orient="records")
        metadata["rows_read"] = len(frame)
    else:
        warnings.append("Experiment metadata table is absent.")
    (out / "metadata.json").write_text(json.dumps(metadata, default=str, indent=2), encoding="utf-8")

    log(f"{dataset}: processing neural events")
    events, channels, audit["events"] = count_records(
        store, f"/{prefix}_wholelife_events", "time_of_event", chunk_rows, freq, True)
    if events.empty:
        raise ValueError("No events have usable timestamps. Inspect the timestamp schema.")
    event_frame = events.rename("detected_events").to_frame()
    rows = [(window, electrode, count) for (window, electrode), count in channels.items()]
    channel_frame = pd.DataFrame(rows, columns=["window_start", "electrode", "detected_events"])
    channel_frame = channel_frame.sort_values(["window_start", "electrode"])
    channel_frame.to_csv(out / "event_electrode_counts.csv", index=False)
    if len(channel_frame):
        grouped = channel_frame.groupby("window_start")
        event_frame["active_observed_electrodes"] = grouped["electrode"].nunique()
        event_frame["electrode_assigned_events"] = grouped["detected_events"].sum()
        event_frame["top_electrode_share"] = (
            grouped["detected_events"].max() / grouped["detected_events"].sum())
    event_frame.to_csv(out / "event_window_features.csv")
    combined = event_frame.copy()

    key = f"/{prefix}_wholelife_stimulations"
    if key in store.keys():
        log(f"{dataset}: processing stimulation records")
        stim, _, audit["stimulation"] = count_records(
            store, key, "time_of_stim", chunk_rows, freq)
        combined = combined.join(stim.rename("stimulation_records"), how="outer")
    else:
        warnings.append("Stimulation table is absent; missing stimulation is not zero.")
        combined["stimulation_records"] = np.nan

    env_rows = []
    for metric, (value_column, output_column) in ENVIRONMENT.items():
        key = f"/{prefix}_wholelife_incubator_{metric}"
        if key not in store.keys():
            warnings.append(f"{metric}: environmental table is absent.")
            continue
        log(f"{dataset}: processing incubator {metric}")
        # Aggregate per incubator and window across chunks, including min/max.
        buckets = {}
        env_audit = {"source_rows": 0, "invalid_time_or_value_rows": 0,
                     "missing_incubator_rows": 0}
        for frame in chunks(store, key, ["time", "incubator_id", value_column], chunk_rows):
            time = timestamps(frame["time"])
            value = pd.to_numeric(frame[value_column], errors="coerce")
            valid = time.notna() & value.notna() & np.isfinite(value)
            missing_id = frame["incubator_id"].isna() | frame["incubator_id"].astype(str).str.strip().eq("")
            env_audit["source_rows"] += len(frame)
            env_audit["invalid_time_or_value_rows"] += int((~valid).sum())
            env_audit["missing_incubator_rows"] += int((valid & missing_id).sum())
            small = pd.DataFrame({"window_start": time.loc[valid].dt.floor(freq),
                                  "incubator_id": frame.loc[valid, "incubator_id"].astype(str).where(~missing_id.loc[valid], "__MISSING__"),
                                  "value": value.loc[valid]})
            stats = small.groupby(["window_start", "incubator_id"])["value"].agg(["sum", "count", "min", "max"])
            for pair, row in stats.iterrows():
                item = buckets.setdefault(pair, [0.0, 0, float("inf"), float("-inf")])
                item[0] += float(row["sum"])
                item[1] += int(row["count"])
                item[2] = min(item[2], float(row["min"]))
                item[3] = max(item[3], float(row["max"]))
        metric_rows = [{"window_start": w, "incubator_id": identifier,
                        "metric": metric, "mean": total / count,
                        "min": minimum, "max": maximum, "observations": count}
                       for (w, identifier), (total, count, minimum, maximum) in buckets.items()]
        env_rows.extend(metric_rows)
        identifiers = {r["incubator_id"] for r in metric_rows}
        env_audit["distinct_incubators"] = len(identifiers)
        audit[metric] = env_audit
        if len(identifiers) == 1 and "__MISSING__" not in identifiers:
            series = pd.Series({r["window_start"]: r["mean"] for r in metric_rows}, name=output_column)
            series.index.name = "window_start"
            combined = combined.join(series, how="outer")
        else:
            warnings.append(f"{metric}: {len(identifiers)} incubator IDs (or missing IDs). Kept separately in environment_by_incubator.csv; not pooled into the main table.")
    env_frame = pd.DataFrame(env_rows, columns=["window_start", "incubator_id", "metric", "mean", "min", "max", "observations"])
    env_frame.to_csv(out / "environment_by_incubator.csv", index=False)

    for column in ["temperature_mean", "co2_mean", "o2_mean"]:
        if column not in combined:
            combined[column] = np.nan
    combined = combined.sort_index()
    grid = pd.date_range(combined.index.min(), combined.index.max(), freq=freq)
    combined = combined.reindex(grid)
    combined.index.name = "window_start"
    combined.to_csv(out / "merged_window_features.csv")
    pd.DataFrame({"available_windows": combined.notna().sum(),
                  "missing_windows": combined.isna().sum()}).to_csv(out / "missingness.csv")

    summary = {
        "dataset": dataset, "window_hours": hours, "total_clock_windows": len(combined),
        "event_rows": audit["events"]["source_rows"],
        "events_with_valid_timestamps": audit["events"]["valid_timestamp_rows"],
        "event_windows": int(combined.detected_events.notna().sum()),
        "stimulation_windows": int(combined.stimulation_records.notna().sum()),
        "first_window_utc": str(combined.index.min()), "last_window_utc": str(combined.index.max()),
        "observed_electrodes": int(channel_frame.electrode.nunique()),
        "metadata_rows_read": metadata.get("rows_read", 0),
        "models_trained": False,
        "interpretation": [
            "Counts describe detected records and are not exposure-normalized firing rates.",
            "Empty windows remain missing; recording coverage is unknown.",
            "Observed electrodes are not independent organoids or a verified channel denominator.",
            "Incubator values retain source units; units and culture mapping require verification.",
            "No age, death, deterioration, or remaining-life labels are inferred.",
            "No spike deduplication or stimulation-artifact exclusion has been applied.",
            "Environmental means are sample-weighted, not time-weighted.",
        ], "warnings": warnings, "audit": audit,
    }
    (out / "analysis_summary.json").write_text(json.dumps(summary, indent=2), encoding="utf-8")
    make_plot(combined, dataset, hours, out)
    return summary


def make_plot(data, dataset, hours, out):
    columns = [("detected_events", "Detected neural events", "Records"),
               ("stimulation_records", "Stimulation records", "Records"),
               ("temperature_mean", "Incubator temperature", "Reported value"),
               ("co2_mean", "Incubator CO2", "Reported value"),
               ("o2_mean", "Incubator O2", "Reported value")]
    fig, axes = plt.subplots(5, 1, figsize=(13, 12), sharex=True, constrained_layout=True)
    for number, (column, title, ylabel) in enumerate(columns):
        ax, series = axes[number], data[column]
        if number < 2:
            valid = series.dropna()
            ax.bar(valid.index, valid.values, width=hours / 24, align="edge", color="#175c71")
            for timestamp in series[series.isna()].index:
                ax.axvspan(timestamp, timestamp + pd.Timedelta(hours=hours), color="gray", alpha=.18)
        elif series.notna().any():
            ax.plot(series.index, series, "o-", color="#175c71", markersize=3)
        else:
            ax.text(.5, .5, "Unavailable or multiple incubators: see separate table", transform=ax.transAxes, ha="center")
        ax.set_title(title, loc="left")
        ax.set_ylabel(ylabel)
        ax.grid(alpha=.15)
    axes[-1].xaxis.set_major_formatter(mdates.DateFormatter("%b %d", tz=timezone.utc))
    axes[-1].set_xlabel("Window start (UTC)")
    axes[-1].set_xlim(data.index.min(), data.index.max() + pd.Timedelta(hours=hours))
    fig.suptitle(f"{dataset}: recorded activity, stimulation and environment")
    fig.supxlabel("Gray = no records in aggregate. Counts are not exposure-normalized.", fontsize=10)
    fig.savefig(out / "combined_overview.png", dpi=160)
    plt.close(fig)


def main():
    parser = argparse.ArgumentParser(description=__doc__, formatter_class=argparse.RawDescriptionHelpFormatter)
    parser.add_argument("--dataset", choices=list(DEFAULT_INPUTS), default="FS369")
    parser.add_argument("--input", type=Path, help="Optional exact package HDF5 path")
    parser.add_argument("--output-root", type=Path, default=DEFAULT_OUTPUT)
    parser.add_argument("--chunk-rows", type=int, default=100_000)
    parser.add_argument("--window-hours", type=int, choices=[1, 2, 3, 4, 6, 8, 12, 24], default=6)
    args = parser.parse_args()
    if args.chunk_rows < 1:
        parser.error("--chunk-rows must be positive")
    source = args.input or DEFAULT_INPUTS[args.dataset]
    if not source.is_file():
        parser.error(f"Input not found: {source}. Supply --input with the correct package path.")
    run_id = datetime.now(timezone.utc).strftime("%Y%m%dT%H%M%S%fZ")
    out = args.output_root / args.dataset / run_id
    out.mkdir(parents=True, exist_ok=False)
    log(f"Input: {source}\nOutput: {out}\nSource files will remain unchanged.")
    try:
        with pd.HDFStore(str(source), mode="r") as store:
            summary = process(store, args.dataset, out, args.chunk_rows, args.window_hours)
        summary["source"] = {"path": str(source.resolve()), "bytes": source.stat().st_size,
                             "modified_ns": source.stat().st_mtime_ns}
        summary["software"] = {"python": platform.python_version(), "pandas": pd.__version__,
                               "numpy": np.__version__, "matplotlib": matplotlib.__version__}
        (out / "analysis_summary.json").write_text(json.dumps(summary, indent=2), encoding="utf-8")
    except Exception as exc:
        (out / "RUN_FAILED.txt").write_text(f"Run incomplete. Do not use as finished output.\n{type(exc).__name__}: {exc}\n", encoding="utf-8")
        raise
    log("\nCOMPLETED")
    for key in ["dataset", "event_rows", "event_windows", "total_clock_windows", "observed_electrodes", "metadata_rows_read"]:
        log(f"{key}: {summary[key]}")
    for warning in summary["warnings"]:
        log(f"NOTE: {warning}")
    log(f"Saved: {out}\nNo raw waveform files opened. No predictive model trained in this step.")


if __name__ == "__main__":
    main()
