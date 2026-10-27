# Maral Farhatnejad | Student ID: 23012856 | BDAPP
"""Descriptive reader for the verified DANDI 002020 conversion (version 0.261004.1410).

Reads detected_spikes DynamicTable, NOT a sorted Units table. No model training,
biological culture-age inference, waveform reads, or FinalSpark data pooling.
"""
from __future__ import annotations
import argparse
from collections import Counter
from datetime import datetime, timezone
import hashlib
import json
from pathlib import Path
import platform
import zipfile
import h5py
import numpy as np
import pandas as pd
import matplotlib
matplotlib.use("Agg")
import matplotlib.pyplot as plt

BASE = "processing/ecephys/"
EVENTS = BASE + "detected_spikes"
PROVENANCE = BASE + "conversion_provenance"
METHODS = [
    "Each record is a recording session, not an independent culture.",
    "Rat dissociated cortical cultures are kept separate from human FinalSpark neurospheres.",
    "The source age P18D, reference gestational, describes the donor at dissection; culture age is unavailable.",
    "recording_elapsed_days is measured from the earliest successfully processed session per culture; it is not days in vitro.",
    "Events are unsorted hardware-channel detections, not identified neurons; multiple detections can represent one neuron.",
    "Relative event times are anchored to the first spike, not the recording start. No absolute spike times are reconstructed.",
    "Count/duration uses converter-reported nominal recording duration, not independently verified continuous exposure.",
    "ISIs and bursts are channel-level event summaries. Burst definition: at least five events joined by ISIs <=100 ms.",
    "Amplitude stays in source-native units. Missing channel mappings remain unresolved.",
    "Duplicate events are retained. No artifact removal, age prediction, survival prediction or causal inference is performed.",
]


def text(value):
    if isinstance(value, bytes):
        return value.decode("utf-8", errors="replace")
    if isinstance(value, np.generic):
        return value.item()
    return value


def scalar(handle, key, default=None):
    if key not in handle:
        return default
    data = handle[key]
    if data.size != 1:
        raise ValueError(f"Expected scalar or length-one value: {key}")
    return text(np.asarray(data[()]).reshape(-1)[0])


def update_channel(state, times):
    """Update pooled ISI moments and fixed-ISI bursts across chunk boundaries."""
    previous = state["last"]
    intervals = np.diff(times if previous is None else np.r_[previous, times])
    if (intervals < 0).any():
        raise ValueError("Channel event times are not chronological")
    state["n_isi"] += len(intervals)
    state["sum_isi"] += float(intervals.sum())
    state["sum_sq_isi"] += float(np.square(intervals).sum())
    state["short_isi"] += int((intervals < .010).sum())
    state["zero_isi"] += int((intervals == 0).sum())
    split = np.flatnonzero(np.diff(times) > .1) + 1
    lengths = np.diff(np.r_[0, split, len(times)]).tolist()
    pending = state["pending"]
    if previous is not None and times[0] - previous <= .1:
        lengths[0] += pending
    elif pending >= 5:
        state["bursts"] += 1
        state["burst_events"] += pending
    for length in lengths[:-1]:
        if length >= 5:
            state["bursts"] += 1
            state["burst_events"] += length
    state["pending"] = lengths[-1]
    state["last"] = float(times[-1])


def inspect_file(path, chunk_rows=100_000):
    states, counts = {}, Counter()
    amp_sum = 0.0
    amp_valid = 0
    first_time = last_time = None
    with h5py.File(path, "r") as f:
        if EVENTS not in f:
            raise ValueError("Verified detected_spikes schema absent; file was not guessed into a Units reader")
        group = f[EVENTS]
        for col in ["source_channel", "time_from_first_spike_seconds"]:
            if col not in group or group[col].ndim != 1:
                raise ValueError(f"Missing one-dimensional event column: {col}")
        times_ds = group["time_from_first_spike_seconds"]
        channels_ds = group["source_channel"]
        n = len(times_ds)
        if len(channels_ds) != n:
            raise ValueError("Channel and event-time lengths differ")
        if channels_ds.dtype.kind not in "iu":
            raise ValueError("Unexpected non-integer hardware-channel identifiers")
        amplitude = group.get("amplitude_native")
        if amplitude is not None and len(amplitude) != n:
            raise ValueError("Amplitude and event-time lengths differ")
        for start in range(0, n, chunk_rows):
            stop = min(n, start + chunk_rows)
            times = np.asarray(times_ds[start:stop], dtype=float)
            channels = np.asarray(channels_ds[start:stop])
            if not np.isfinite(times).all() or (times < 0).any():
                raise ValueError("Invalid first-spike-relative times")
            if (np.diff(times) < 0).any() or (last_time is not None and times[0] < last_time):
                raise ValueError("Global event order is not chronological; timing features withheld")
            if first_time is None:
                first_time = float(times[0])
            last_time = float(times[-1])
            if amplitude is not None:
                values = np.asarray(amplitude[start:stop], dtype=float)
                values = values[np.isfinite(values)]
                amp_sum += float(np.abs(values).sum())
                amp_valid += len(values)
            # Stable grouping preserves chronological order within each channel.
            order = np.argsort(channels, kind="stable")
            sorted_channels, sorted_times = channels[order], times[order]
            boundaries = np.r_[0, np.flatnonzero(np.diff(sorted_channels) != 0) + 1, len(times)]
            for lo, hi in zip(boundaries[:-1], boundaries[1:]):
                channel = int(sorted_channels[lo])
                values = sorted_times[lo:hi]
                counts[channel] += len(values)
                state = states.setdefault(channel, {"last": None, "pending": 0,
                    "bursts": 0, "burst_events": 0, "n_isi": 0,
                    "sum_isi": 0.0, "sum_sq_isi": 0.0, "short_isi": 0, "zero_isi": 0})
                update_channel(state, values)
        for state in states.values():
            if state["pending"] >= 5:
                state["bursts"] += 1
                state["burst_events"] += state["pending"]
        duration = scalar(f, PROVENANCE + "/duration_s")
        duration = float(duration) if duration is not None else np.nan
        if not np.isnan(duration) and (not np.isfinite(duration) or duration <= 0):
            raise ValueError("Invalid declared recording duration")
        if last_time is not None and np.isfinite(duration) and last_time - first_time > duration + 1e-6:
            raise ValueError("Observed event span exceeds declared recording duration")
        ni = sum(s["n_isi"] for s in states.values())
        mean = sum(s["sum_isi"] for s in states.values()) / ni if ni else np.nan
        moment = sum(s["sum_sq_isi"] for s in states.values()) / ni if ni else np.nan
        cv = np.sqrt(max(0, moment - mean ** 2)) / mean if ni and mean > 0 else np.nan
        bursts = sum(s["bursts"] for s in states.values())
        row = {
            "path": str(path), "recording_id": scalar(f, "identifier"),
            "culture_id": str(scalar(f, "general/subject/subject_id", "")),
            "session_id": scalar(f, "general/session_id"),
            "session_start": scalar(f, "session_start_time"),
            "species": scalar(f, "general/subject/species"),
            "source_donor_age": scalar(f, "general/subject/age"),
            "source_age_reference": text(f["general/subject/age"].attrs.get("reference", "")) if "general/subject/age" in f else "",
            "culture_age_days": np.nan,
            "nominal_duration_s": duration, "detected_events": n,
            "active_source_channels": len(counts),
            "events_per_nominal_recording_second": n / duration if np.isfinite(duration) else np.nan,
            "observed_event_span_seconds": last_time - first_time if first_time is not None else np.nan,
            "channel_isi_count": ni, "pooled_channel_isi_mean_s": mean,
            "pooled_channel_isi_cv": cv,
            "fraction_channel_intervals_under_10ms": sum(s["short_isi"] for s in states.values()) / ni if ni else np.nan,
            "zero_channel_intervals": sum(s["zero_isi"] for s in states.values()),
            "channel_burst_count": bursts,
            "fraction_events_in_channel_bursts": sum(s["burst_events"] for s in states.values()) / n if n else np.nan,
            "mean_abs_amplitude_native": amp_sum / amp_valid if amp_valid else np.nan,
            "amplitude_valid_events": amp_valid,
            "amplitude_unit": scalar(f, PROVENANCE + "/amplitude_unit"),
            "source_unresolved_channel_count": scalar(f, PROVENANCE + "/unresolved_channel_count"),
            "source_unresolved_event_count": scalar(f, PROVENANCE + "/unresolved_event_count"),
        }
        if not row["culture_id"] or not row["recording_id"] or not row["session_start"]:
            raise ValueError("Required culture/session identity is absent")
        prov = {key: scalar(f, PROVENANCE + "/" + key) for key in [
            "converter_version", "sampling_rate_hz", "sampling_rate_basis", "duration_s",
            "timestamp_method", "recording_frame_origin", "source_timestamp_interpretation"]}
        evidence = {"path": str(path), "nwb_version": text(f.attrs.get("nwb_version", "")),
                    "subject_description": scalar(f, "general/subject/description"),
                    "experiment_description": scalar(f, "general/experiment_description"),
                    "event_description": text(group.attrs.get("description", "")), "provenance": prov}
    if sum(counts.values()) != n or ni != n - len(counts):
        raise AssertionError("Event/channel/ISI count consistency failed")
    return row, evidence


def run(root, out, limit=0, chunk_rows=100_000):
    paths = sorted(root.rglob("*.nwb"))
    if not paths:
        raise FileNotFoundError(f"No .nwb files beneath {root}")
    selected = paths[:limit] if limit else paths
    out.mkdir(parents=True, exist_ok=False)
    rows, audit, evidence, seen_ids, seen_sessions = [], [], [], set(), set()
    for number, path in enumerate(selected, 1):
        print(f"[{number}/{len(selected)}] {path.name}", flush=True)
        try:
            row, info = inspect_file(path, chunk_rows)
            session_key = (row["culture_id"], row["session_start"])
            if row["recording_id"] in seen_ids or session_key in seen_sessions:
                raise ValueError("Duplicate recording ID or culture/session time; duplicate excluded")
            parsed = pd.Timestamp(row["session_start"])
            if parsed.tzinfo is None:
                raise ValueError("Session timestamp lacks timezone")
            seen_ids.add(row["recording_id"])
            seen_sessions.add(session_key)
            row["session_start_utc"] = parsed.tz_convert("UTC").isoformat()
            rows.append(row)
            evidence.append(info)
            audit.append({"path": str(path), "status": "processed", "message": ""})
        except Exception as exc:
            audit.append({"path": str(path), "status": "skipped", "message": f"{type(exc).__name__}: {exc}"})
            print("  SKIPPED:", exc, flush=True)
    data = pd.DataFrame(rows)
    if len(data):
        dates = pd.to_datetime(data.session_start_utc, utc=True)
        first_dates = dates.groupby(data.culture_id).transform("min")
        data["recording_elapsed_days"] = (dates - first_dates).dt.total_seconds() / 86400
        data = data.sort_values(["culture_id", "session_start_utc"])
        data.to_csv(out / "dandi_recording_features.csv", index=False)
        data.groupby("culture_id").agg(recordings=("session_id", "size"),
            first_session_utc=("session_start_utc", "min"), last_session_utc=("session_start_utc", "max"),
            detected_events=("detected_events", "sum")).to_csv(out / "dandi_culture_summary.csv")
        fig, axes = plt.subplots(2, 1, figsize=(11, 8), sharex=True, constrained_layout=True)
        for culture, group in data.groupby("culture_id"):
            axes[0].plot(group.recording_elapsed_days, group.events_per_nominal_recording_second, "o-", label=culture, markersize=3)
            axes[1].plot(group.recording_elapsed_days, group.fraction_events_in_channel_bursts, "o-", label=culture, markersize=3)
        axes[0].set_ylabel("Events / nominal recording second")
        axes[0].legend(title="Culture", ncol=4, fontsize=8)
        axes[1].set_ylabel("Fraction of events in channel bursts")
        axes[1].set_xlabel("Days since first successfully processed recording per culture")
        fig.suptitle("DANDI rat cultures: descriptive event activity")
        fig.savefig(out / "dandi_activity_overview.png", dpi=160)
        plt.close(fig)
    else:
        pd.DataFrame(columns=["culture_id", "session_id"]).to_csv(out / "dandi_recording_features.csv", index=False)
    pd.DataFrame(audit).to_csv(out / "dandi_file_audit.csv", index=False)
    (out / "dandi_metadata_evidence.json").write_text(json.dumps(evidence, indent=2), encoding="utf-8")
    status = {"discovered_files": len(paths), "selected_files": len(selected), "processed_files": len(rows),
              "skipped_files": len(selected) - len(rows), "cultures": int(data.culture_id.nunique()) if len(data) else 0,
              "culture_age_prediction": "not performed: culture age unavailable", "FinalSpark_model_validation": "not performed",
              "methods_and_limitations": METHODS,
              "software": {"python": platform.python_version(), "h5py": h5py.__version__, "numpy": np.__version__, "pandas": pd.__version__}}
    (out / "dandi_analysis_status.json").write_text(json.dumps(status, indent=2), encoding="utf-8")
    bundle = out / "DANDI_summary.zip"
    with zipfile.ZipFile(bundle, "w", zipfile.ZIP_DEFLATED) as archive:
        for path in sorted(out.iterdir()):
            if path.is_file() and path != bundle:
                archive.write(path, path.name)
    print("\nDANDI COMPLETE")
    print(f"Processed: {len(rows)}/{len(selected)} files; cultures: {status['cultures']}")
    print("Upload this derived-results ZIP:", bundle)
    if not rows:
        raise RuntimeError("No files processed. Upload the audit ZIP so the actual schema can be checked.")
    return data, status


if __name__ == "__main__":
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--root", type=Path, default=Path(r"F:\DANDI\002020-0.261004.1410"))
    parser.add_argument("--output", type=Path)
    parser.add_argument("--limit", type=int, default=0)
    parser.add_argument("--chunk-rows", type=int, default=100_000)
    args = parser.parse_args()
    if args.limit < 0 or args.chunk_rows < 1:
        parser.error("--limit must be >=0 and --chunk-rows must be positive")
    output = args.output or Path(r"F:\BDAPP-Organoid\outputs\DANDI") / datetime.now(timezone.utc).strftime("%Y%m%dT%H%M%S%fZ")
    run(args.root, output, args.limit, args.chunk_rows)
