# Maral Farhatnejad | Student ID: 23012856 | BDAPP
"""Read Zenodo 22755004 individual organoid MAT files, extracted or zipped."""
from pathlib import Path
from datetime import datetime, timezone
import argparse, hashlib, io, json, re, zipfile
import h5py
import numpy as np
import pandas as pd
import matplotlib
matplotlib.use("Agg")
import matplotlib.pyplot as plt
from hourly_stage import RECORD, features

PATTERN = re.compile(r"^fs\d+-org\d+\.mat$", re.I)
NOTES = [
    "Separate descriptive human forebrain organoid dataset; not pooled with whole-life FinalSpark models.",
    "Durations come from each file's metadata/recording_duration, not the publication-wide duration description or last spike.",
    "Per-organoid age and independent death labels are not supplied in the verified MAT schema. No age or lifespan model is evaluated.",
    "The cohort age range from the listing is not assigned to individual recordings.",
    "Rates use declared duration and channel count. Continuous exposure has not been independently checked.",
    "Channel interval/burst definitions match the implemented fixed-ISI extractor, not all MATLAB publication analyses.",
    "Burst threshold: five events connected by ISIs <=100 ms. Bursts are censored at recording boundaries.",
    "Amplitude retains source units. Neuron identity, intelligence and causal effects are not inferred.",
    "If a channel's amplitude array length differs from its spike train, its amplitudes are withheld and the mismatch is reported; valid event times remain included.",
    "Organoids sharing an fs experiment prefix may be related. File count is not verified independent biological replication.",
]


def matlab_text(dataset):
    return "".join(chr(int(x)) for x in np.asarray(dataset).reshape(-1) if x)


def read_mat(content, source):
    with h5py.File(io.BytesIO(content), "r") as f:
        if not {"metadata", "spike_times", "organoid_id"}.issubset(f):
            raise ValueError("Unsupported MAT schema")
        meta = f["metadata"]
        duration = float(np.asarray(meta["recording_duration"]).ravel()[0])
        channels = np.asarray(meta["channel_ids"]).ravel()
        if not np.isfinite(duration) or duration <= 0:
            raise ValueError("Invalid file-specific duration")
        if not np.isfinite(channels).all() or not np.equal(channels, np.floor(channels)).all() or len(set(channels)) != len(channels):
            raise ValueError("Invalid channel IDs")
        refs = np.asarray(f["spike_times"]).ravel()
        if len(refs) != len(channels):
            raise ValueError("Channel count and spike-cell count differ")
        amplitude_refs = np.asarray(f["amplitudes"]).ravel() if "amplitudes" in f else None
        chunks, amplitude_mismatches = [], []
        for i, (channel, ref) in enumerate(zip(channels, refs)):
            obj = f[ref]
            t = np.array([]) if obj.attrs.get("MATLAB_empty", False) else np.asarray(obj, dtype=float).ravel()
            if not np.isfinite(t).all() or (t < 0).any() or (t >= duration).any():
                raise ValueError("Spike times contradict declared duration in seconds")
            amp = np.full(len(t), np.nan)
            if amplitude_refs is not None:
                a = f[amplitude_refs[i]]
                a = np.array([]) if a.attrs.get("MATLAB_empty", False) else np.asarray(a, dtype=float).ravel()
                if len(a) != len(t):
                    amplitude_mismatches.append(int(channel))
                else:
                    amp = a
            array = np.empty(len(t), dtype=RECORD)
            array["time_ns"] = np.rint(t * 1e9).astype(np.int64)
            array["channel"] = int(channel)
            array["amplitude"] = amp
            chunks.append(array)
        events = np.concatenate(chunks) if chunks else np.empty(0, dtype=RECORD)
        oid = matlab_text(f["organoid_id"])
        stats = features(events) if len(events) else {"detected_events": 0, "active_observed_channels": 0}
        # The shared extractor's voltage suffix is deliberately removed here:
        # the public MAT amplitude units were not independently established.
        for key in list(stats):
            if key.endswith("_uv"):
                stats[key[:-3] + "_source_units"] = stats.pop(key)
        stats.update({"organoid_id": oid, "experiment_prefix": oid.split("-org")[0],
            "source": source, "sha256": hashlib.sha256(content).hexdigest(),
            "declared_duration_s": duration, "declared_channels": len(channels),
            "amplitude_mismatch_channel_count": len(amplitude_mismatches),
            "amplitude_mismatch_channels": ";".join(map(str, amplitude_mismatches)),
            "events_per_declared_second": len(events) / duration,
            "mean_events_per_declared_channel_second": len(events) / (len(channels) * duration) if len(channels) else np.nan,
            "biological_age_days": np.nan, "remaining_life_days": np.nan,
            "metadata_keys": ";".join(meta.keys())})
        if "fs_id" in meta and "org_id" in meta:
            expected = "fs%d-org%d" % (int(np.asarray(meta["fs_id"]).ravel()[0]), int(np.asarray(meta["org_id"]).ravel()[0]))
            if oid != expected:
                raise ValueError("Organoid ID conflicts with metadata IDs")
        return stats


def sources(root):
    for p in sorted(root.rglob("*.mat")):
        if PATTERN.match(p.name):
            yield str(p), p.read_bytes()
    for p in sorted(root.rglob("*.zip")):
        if p.name.lower() != "indiv_organoids.zip":
            continue
        with zipfile.ZipFile(p) as z:
            for name in sorted(z.namelist()):
                if PATTERN.match(Path(name).name):
                    yield str(p) + "::" + name, z.read(name)


def run(root, out):
    out.mkdir(parents=True, exist_ok=False)
    rows, audit, seen = [], [], {}
    for number, (name, data) in enumerate(sources(root), 1):
        print(f"[{number}] {name}", flush=True)
        try:
            row = read_mat(data, name)
            oid = row["organoid_id"]
            if oid in seen:
                status = "duplicate_identical" if row["sha256"] == seen[oid] else "CONFLICT_DIFFERENT_CONTENT"
                audit.append({"source": name, "status": status, "message": oid})
                continue
            seen[oid] = row["sha256"]
            rows.append(row)
            audit.append({"source": name, "status": "processed", "message": ""})
        except Exception as exc:
            audit.append({"source": name, "status": "skipped", "message": f"{type(exc).__name__}: {exc}"})
    if not audit:
        raise FileNotFoundError("No individual fs*-org*.mat files or Indiv_organoids.zip found under " + str(root))
    d = pd.DataFrame(rows)
    d.to_csv(out / "zenodo_organoid_features.csv", index=False)
    pd.DataFrame(audit).to_csv(out / "zenodo_file_audit.csv", index=False)
    status = {"processed_organoids": len(rows), "experiment_prefixes": int(d.experiment_prefix.nunique()) if len(d) else 0,
        "detected_events": int(d.detected_events.sum()) if len(d) else 0,
        "status_counts": pd.Series([x["status"] for x in audit]).value_counts().to_dict(),
        "duration_values_s": sorted(d.declared_duration_s.unique().tolist()) if len(d) else [],
        "biological_age_prediction": "not performed: no per-organoid labels", "lifespan_prediction": "not performed",
        "notes": NOTES}
    (out / "zenodo_analysis_status.json").write_text(json.dumps(status, indent=2), encoding="utf-8")
    if len(d):
        fig, axes = plt.subplots(1, 2, figsize=(11, 4), constrained_layout=True)
        axes[0].hist(d.mean_events_per_declared_channel_second, bins=15, color="#216b74")
        axes[0].set_xlabel("Mean events per declared channel-second")
        axes[0].set_ylabel("Organoid recordings")
        axes[1].hist(d.fraction_events_in_channel_bursts.dropna(), bins=15, color="#216b74")
        axes[1].set_xlabel("Fraction of events in fixed-ISI channel bursts")
        fig.suptitle("Zenodo human forebrain organoids: descriptive activity")
        fig.savefig(out / "zenodo_activity_overview.png", dpi=170)
        plt.close(fig)
    bundle = out / "Zenodo_summary.zip"
    with zipfile.ZipFile(bundle, "w", zipfile.ZIP_DEFLATED) as z:
        for p in sorted(out.iterdir()):
            if p.is_file() and p != bundle:
                z.write(p, p.name)
    print("\nZENODO COMPLETE")
    print(json.dumps(status, indent=2))
    print("Upload:", bundle)
    if not rows or any(x["status"] == "CONFLICT_DIFFERENT_CONTENT" for x in audit):
        raise RuntimeError("No processed rows or conflicting organoid copies. Upload the audit ZIP.")
    return d, status


if __name__ == "__main__":
    p = argparse.ArgumentParser(description=__doc__)
    p.add_argument("--root", type=Path, default=Path(r"F:\DANDI\22755004"))
    p.add_argument("--output", type=Path)
    a = p.parse_args()
    out = a.output or Path(r"F:\BDAPP-Organoid\outputs\Zenodo") / datetime.now(timezone.utc).strftime("%Y%m%dT%H%M%S%fZ")
    run(a.root, out)
