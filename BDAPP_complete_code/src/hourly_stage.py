# Maral Farhatnejad | Student ID: 23012856 | BDAPP
from pathlib import Path
from datetime import datetime, timezone
import argparse
import json
import shutil
import zipfile
import numpy as np
import pandas as pd

INPUTS = {
    "FS369": Path(r"F:\FinalSpark\fs369data\fs369_package.hdf5"),
    "FS437": Path(r"F:\FinalSpark\fs437\fs437_export\fs437_package.hdf5"),
}
RECORD = np.dtype([("time_ns", "<i8"), ("channel", "<i8"), ("amplitude", "<f8")])
HOUR_NS = 3_600_000_000_000


def utc(series):
    if pd.api.types.is_numeric_dtype(series.dtype):
        raise ValueError("Numeric timestamp units require verification")
    return pd.to_datetime(series, errors="coerce", utc=True)


def stream(store, key, columns):
    if key not in store.keys() or not store.get_storer(key).is_table:
        raise ValueError(f"Missing chunk-readable table: {key}")
    return store.select(key, columns=columns, chunksize=100_000, iterator=True)


def features(records):
    result = {"detected_events": len(records)}
    order = np.lexsort((records["time_ns"], records["channel"]))
    r = records[order]
    breaks = np.r_[0, np.flatnonzero(np.diff(r["channel"]) != 0) + 1, len(r)]
    counts = np.diff(breaks)
    result["active_observed_channels"] = len(counts)
    probs = counts / counts.sum()
    result["channel_count_entropy"] = float(-(probs * np.log(probs)).sum())
    result["largest_channel_event_share"] = float(probs.max())
    amp = np.abs(r["amplitude"][np.isfinite(r["amplitude"])])
    result["valid_amplitude_events"] = len(amp)
    result["amplitude_abs_median_uv"] = float(np.median(amp)) if len(amp) else np.nan
    result["amplitude_abs_iqr_uv"] = float(np.percentile(amp, 75) - np.percentile(amp, 25)) if len(amp) else np.nan
    intervals, cv2_values = [], []
    bursts = burst_events = duplicate_candidates = 0
    for lo, hi in zip(breaks[:-1], breaks[1:]):
        times = r["time_ns"][lo:hi]
        isi = np.diff(times) / 1e9
        duplicate_candidates += int((isi == 0).sum())
        if len(isi):
            intervals.append(isi)
        if len(isi) > 1:
            denom = isi[1:] + isi[:-1]
            positive = denom > 0
            cv2_values.append(2 * np.abs(isi[1:][positive] - isi[:-1][positive]) / denom[positive])
        lengths = np.diff(np.r_[0, np.flatnonzero(isi > .1) + 1, hi - lo])
        qualifying = lengths[lengths >= 5]
        bursts += len(qualifying)
        burst_events += int(qualifying.sum())
    isi = np.concatenate(intervals) if intervals else np.array([])
    cv2 = np.concatenate(cv2_values) if cv2_values else np.array([])
    result.update({
        "channel_interval_count": len(isi),
        "channel_interval_median_s": float(np.median(isi)) if len(isi) else np.nan,
        "channel_interval_cv": float(isi.std() / isi.mean()) if len(isi) and isi.mean() > 0 else np.nan,
        "channel_interval_cv2": float(cv2.mean()) if len(cv2) else np.nan,
        "fraction_channel_intervals_under_10ms": float((isi < .010).mean()) if len(isi) else np.nan,
        "same_channel_same_time_candidates": duplicate_candidates,
        "channel_burst_count": bursts,
        "fraction_events_in_channel_bursts": burst_events / len(r),
        "verified_recording_seconds": np.nan,
        "exposure_normalized_rate_hz": np.nan,
    })
    return result


def process(source, experiment, folder, max_events):
    if not source.is_file():
        raise FileNotFoundError(source)
    folder.mkdir(parents=True, exist_ok=False)
    spool = folder / "temporary_hour_partitions"
    spool.mkdir()
    prefix = experiment.lower()
    stats = {"source_rows": 0, "invalid_time_or_channel_rows": 0, "hours_written": 0}
    with pd.HDFStore(str(source), "r") as store:
        key = f"/{prefix}_wholelife_events"
        total = store.get_storer(key).nrows
        needed = int(total) * RECORD.itemsize + 256 * 1024 ** 2
        if shutil.disk_usage(folder).free < needed:
            raise OSError(f"Need approximately {needed / 1024**3:.2f} GiB free for temporary event partitions")
        for number, part in enumerate(stream(store, key, ["time_of_event", "electrode", "max_voltage_uv"]), 1):
            dates = utc(part.time_of_event)
            channels = pd.to_numeric(part.electrode, errors="coerce")
            valid = dates.notna() & channels.notna() & np.isfinite(channels) & channels.eq(np.floor(channels))
            stats["source_rows"] += len(part)
            stats["invalid_time_or_channel_rows"] += int((~valid).sum())
            ns = dates.loc[valid].dt.as_unit("ns").astype("int64").to_numpy()
            hour = ns // HOUR_NS
            a = np.empty(len(ns), dtype=RECORD)
            a["time_ns"] = ns
            a["channel"] = channels.loc[valid].to_numpy(dtype=np.int64)
            a["amplitude"] = pd.to_numeric(part.loc[valid, "max_voltage_uv"], errors="coerce").to_numpy(dtype=float)
            order = np.argsort(hour, kind="stable")
            hour, a = hour[order], a[order]
            cuts = np.r_[0, np.flatnonzero(np.diff(hour) != 0) + 1, len(hour)]
            for lo, hi in zip(cuts[:-1], cuts[1:]):
                if hi > lo:
                    with (spool / f"{int(hour[lo])}.bin").open("ab") as handle:
                        a[lo:hi].tofile(handle)
            if number == 1 or number % 20 == 0:
                print(f"{experiment}: partitioned {stats['source_rows']:,}/{total:,} events", flush=True)
        feature_rows = []
        for number, path in enumerate(sorted(spool.glob("*.bin"), key=lambda p: int(p.stem)), 1):
            n = path.stat().st_size // RECORD.itemsize
            start = pd.Timestamp(int(path.stem) * HOUR_NS, tz="UTC")
            row = {"window_start": start, "assessment_utc": start + pd.Timedelta(hours=1)}
            if n > max_events:
                row.update({"detected_events": n, "feature_status": "count_only_memory_guard"})
            else:
                row.update(features(np.fromfile(path, dtype=RECORD)))
                row["feature_status"] = "event_features_extracted_coverage_unknown"
            feature_rows.append(row)
            if number == 1 or number % 100 == 0:
                print(f"{experiment}: extracted {number} event-containing hours", flush=True)
        if not feature_rows:
            raise ValueError("No event-containing hours")
        stats["hours_written"] = len(feature_rows)
        result = pd.DataFrame(feature_rows).set_index("window_start")
        stim_counts = {}
        key = f"/{prefix}_wholelife_stimulations"
        if key in store.keys():
            for part in stream(store, key, ["time_of_stim"]):
                dates = utc(part.time_of_stim)
                for t, n in dates.dropna().dt.floor("h").value_counts().items():
                    stim_counts[t] = stim_counts.get(t, 0) + int(n)
            result["export_stimulation_records"] = [stim_counts.get(t, 0) for t in result.index]
        else:
            result["export_stimulation_records"] = np.nan
        for metric, col in [("temperature", "temperature_mean"), ("CO2", "co2_mean"), ("O2", "o2_mean")]:
            key = f"/{prefix}_wholelife_incubator_{metric}"
            sums, nums, identifiers = {}, {}, set()
            if key in store.keys():
                for part in stream(store, key, ["time", "incubator_id", f"incubator_{metric}"]):
                    dates = utc(part.time)
                    values = pd.to_numeric(part[f"incubator_{metric}"], errors="coerce")
                    valid = dates.notna() & values.notna() & np.isfinite(values)
                    identifiers.update(part.loc[valid, "incubator_id"].fillna("__MISSING__").astype(str))
                    small = pd.DataFrame({"hour": dates[valid].dt.floor("h"), "value": values[valid]})
                    for t, group in small.groupby("hour").value:
                        sums[t] = sums.get(t, 0.) + float(group.sum())
                        nums[t] = nums.get(t, 0) + len(group)
            if len(identifiers) == 1 and "__MISSING__" not in identifiers:
                result[col] = pd.Series({t: sums[t] / nums[t] for t in sums})
            else:
                result[col] = np.nan
            stats[metric + "_incubator_ids"] = sorted(identifiers)
        meta = store.get(f"/{prefix}_wholelife_metadata")
        metadata = json.loads(meta.to_json(orient="records", date_format="iso", default_handler=str))
        if len(metadata) != 1:
            raise ValueError("Expected one experiment metadata record")
        birth = pd.Timestamp(metadata[0]["Batch Start"], tz="UTC")
        endpoint = pd.Timestamp(metadata[0]["Dead Date"], tz="UTC")
        recording_start = pd.Timestamp(metadata[0]["Date"], tz="UTC")
    result = result.sort_index().reindex(pd.date_range(result.index.min(), result.index.max(), freq="h"))
    result.index.name = "window_start"
    result["assessment_utc"] = result.index + pd.Timedelta(hours=1)
    result["feature_status"] = result.feature_status.fillna("no_exported_events_coverage_unknown")
    result["experiment_id"] = experiment
    result["batch_calendar_age_days"] = (result.assessment_utc - birth).dt.total_seconds() / 86400
    result["days_until_recorded_experiment_endpoint"] = (endpoint - result.assessment_utc).dt.total_seconds() / 86400
    result["window_before_recorded_endpoint"] = result.assessment_utc.le(endpoint)
    result["eligible_for_exploratory_model"] = (result.feature_status.eq("event_features_extracted_coverage_unknown")
        & result.assessment_utc.le(endpoint) & pd.Series(result.index >= recording_start, index=result.index))
    if int(result.detected_events.sum()) != stats["source_rows"] - stats["invalid_time_or_channel_rows"]:
        raise AssertionError("Hourly counts disagree with input row audit")
    result.to_csv(folder / "hourly_features.csv")
    (folder / "metadata.json").write_text(json.dumps(metadata, indent=2), encoding="utf-8")
    stats.update({"clock_hours": len(result), "event_hours": int(result.detected_events.notna().sum()),
                  "eligible_hours": int(result.eligible_for_exploratory_model.sum()),
                  "memory_guard_hours": int(result.feature_status.eq("count_only_memory_guard").sum())})
    (folder / "extraction_audit.json").write_text(json.dumps(stats, indent=2), encoding="utf-8")
    shutil.rmtree(spool)
    print(experiment, stats, flush=True)
    return result


def main():
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--output", type=Path)
    parser.add_argument("--max-window-events", type=int, default=5_000_000)
    parser.add_argument("--fs369", type=Path, default=INPUTS["FS369"])
    parser.add_argument("--fs437", type=Path, default=INPUTS["FS437"])
    args = parser.parse_args()
    out = args.output or Path(r"F:\BDAPP-Organoid\outputs\hourly") / datetime.now(timezone.utc).strftime("%Y%m%dT%H%M%S%fZ")
    out.mkdir(parents=True, exist_ok=False)
    try:
        for experiment, source in {"FS369": args.fs369, "FS437": args.fs437}.items():
            process(source, experiment, out / experiment, args.max_window_events)
        notes = [
            "One-hour UTC clock windows; all timing features use only events inside that window.",
            "Timing is sorted within channel and window; chunk input order is not assumed.",
            "Channel bursts: at least five events linked by intervals <=100 ms. Window edges censor bursts.",
            "Coverage remains unknown. Rates, physiological silence and true spontaneous activity are not established.",
            "No events were deduplicated or removed as stimulation artifacts. No culture mapping was guessed.",
            "Export stimulation counts indicate log records only; no logged record does not verify no stimulation.",
            "Amplitude uses the documented max_voltage_uv field. Environmental units remain source-reported.",
            "Age/endpoint labels are the same exploratory experiment-level targets as the six-hour baseline.",
            "This file contains extraction outputs only. Run hourly_models.py separately for exploratory comparisons.",
        ]
        (out / "METHODS.txt").write_text("\n".join(notes), encoding="utf-8")
        bundle = out / "FinalSpark_hourly_summary.zip"
        with zipfile.ZipFile(bundle, "w", zipfile.ZIP_DEFLATED) as archive:
            for path in out.rglob("*"):
                if path.is_file() and path != bundle:
                    archive.write(path, path.relative_to(out))
        print("\nHOURLY EXTRACTION COMPLETE\nUpload:", bundle, flush=True)
    except Exception as exc:
        (out / "RUN_FAILED.txt").write_text(f"Run incomplete: {type(exc).__name__}: {exc}", encoding="utf-8")
        raise


if __name__ == "__main__":
    main()
