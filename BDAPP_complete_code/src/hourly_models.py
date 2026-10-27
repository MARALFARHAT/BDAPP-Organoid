# Maral Farhatnejad | Student ID: 23012856 | BDAPP
from pathlib import Path
from datetime import datetime, timezone
import argparse, hashlib, io, json, platform, zipfile
import numpy as np
import pandas as pd
import sklearn
from sklearn.dummy import DummyRegressor
from sklearn.ensemble import RandomForestRegressor
from sklearn.impute import SimpleImputer
from sklearn.linear_model import Ridge
from sklearn.pipeline import make_pipeline
from sklearn.preprocessing import StandardScaler
from sklearn.metrics import mean_absolute_error, mean_squared_error, r2_score
import matplotlib
matplotlib.use("Agg")
import matplotlib.pyplot as plt

TARGETS = ["batch_calendar_age_days", "days_until_recorded_experiment_endpoint"]
GROUPS = {
    "counts": ["log_event_count", "active_observed_channels"],
    "distribution": ["channel_count_entropy", "largest_channel_event_share"],
    "amplitude": ["amplitude_abs_median_uv", "amplitude_abs_iqr_uv"],
    "timing": ["channel_interval_median_s", "channel_interval_cv", "channel_interval_cv2", "fraction_channel_intervals_under_10ms"],
    "bursts": ["log_burst_count", "fraction_events_in_channel_bursts"],
}
NEURAL = [c for group in GROUPS.values() for c in group]
ENV = ["temperature_mean", "co2_mean", "o2_mean"]
SPECS = {
    "training_median": (["log_event_count"], "median"),
    "count_only_ridge": (["log_event_count"], "ridge"),
    "rich_neural_ridge": (NEURAL, "ridge"),
    "rich_neural_forest": (NEURAL, "forest"),
    "environment_only_ridge": (ENV, "ridge"),
    "neural_environment_ridge": (NEURAL + ENV, "ridge"),
    "neural_environment_forest": (NEURAL + ENV, "forest"),
    "neural_environment_stimulation_ridge": (NEURAL + ENV + ["log_export_stimulation"], "ridge"),
}
LIMITS = [
    "Two aggregate experiments share the same reported batch and creation time. Independent biological replication is unverified.",
    "Hourly windows are dependent observations, not independent organoids. No window-bootstrap confidence intervals are reported.",
    "Age is calendar time from the documented Batch Start, not independently measured maturation.",
    "Endpoint labels are experiment-level and activity-defined. They are not independently verified individual deaths.",
    "Exposure is unknown. Counts are not firing rates; channel timing and bursts can be affected by recording gaps and detection conditions.",
    "Zero exported stimulation records does not establish spontaneous activity. The sensitivity check excludes only hours with logged triggers, without a pre/post-stimulation guard.",
    "Amplitude is the documented max_voltage_uv field, with no artifact rejection. It is not a waveform-based health measure.",
    "Train/test ages and environmental measurements differ markedly. All FS437 results are exploratory after earlier inspection of this experiment.",
    "All configurations and ablations are retained. There is no test-set tuning, clipping, selected winner, or population validation claim.",
    "DANDI and Zenodo are separate descriptive tracks; neither provides validation of these age/endpoint models.",
]


def prepare(z, experiment):
    d = pd.read_csv(io.BytesIO(z.read(experiment + "/hourly_features.csv")))
    audit = json.loads(z.read(experiment + "/extraction_audit.json"))
    m = json.loads(z.read(experiment + "/metadata.json"))[0]
    d["window_start"] = pd.to_datetime(d.window_start, utc=True)
    d["assessment_utc"] = pd.to_datetime(d.assessment_utc, utc=True)
    if d.window_start.duplicated().any() or not d.assessment_utc.eq(d.window_start + pd.Timedelta(hours=1)).all():
        raise ValueError("Duplicate windows or incorrect assessment times")
    if int(d.detected_events.sum()) != audit["source_rows"] - audit["invalid_time_or_channel_rows"]:
        raise ValueError("Source/event-count audit mismatch")
    birth, start, endpoint = [pd.Timestamp(m[k], tz="UTC") for k in ["Batch Start", "Date", "Dead Date"]]
    actual_labels = [(d.assessment_utc - birth).dt.total_seconds() / 86400,
                     (endpoint - d.assessment_utc).dt.total_seconds() / 86400]
    for target, actual in zip(TARGETS, actual_labels):
        if not np.allclose(d[target], actual):
            raise ValueError("Prepared target and metadata disagree")
        d[target] = actual
    eligible = (d.feature_status.eq("event_features_extracted_coverage_unknown")
                & d.detected_events.notna() & d.window_start.ge(start) & d.assessment_utc.le(endpoint))
    d["eligible_checked"] = eligible
    d["experiment_id"] = experiment
    d["log_event_count"] = np.log1p(d.detected_events)
    d["log_burst_count"] = np.log1p(d.channel_burst_count)
    d["log_export_stimulation"] = np.log1p(d.export_stimulation_records)
    for col in NEURAL + ENV + ["log_export_stimulation"]:
        if not np.isfinite(d[col].dropna()).all():
            raise ValueError("Non-finite feature: " + col)
    if d.loc[eligible, NEURAL].isna().any().any():
        raise ValueError("Missing neural features in eligible hours")
    return d.sort_values("window_start").reset_index(drop=True)


def estimator(kind):
    if kind == "median":
        model = DummyRegressor(strategy="median")
    elif kind == "forest":
        model = RandomForestRegressor(n_estimators=200, max_depth=4, min_samples_leaf=5, random_state=42, n_jobs=1)
    else:
        model = Ridge(alpha=10.0)
    return make_pipeline(SimpleImputer(strategy="median", add_indicator=True), StandardScaler(), model)


def evaluate(train, test, protocol, ablations=False):
    specs = dict(SPECS)
    if ablations:
        for group, removed in GROUPS.items():
            specs["ablation_without_" + group] = ([c for c in NEURAL if c not in removed], "ridge")
    scores, predictions = [], []
    if len(train) < 20 or len(test) < 5:
        raise ValueError("Insufficient rows for specified evaluation")
    for target in TARGETS:
        y = test[target].to_numpy()
        for name, (columns, kind) in specs.items():
            if train[columns].notna().sum().eq(0).any():
                raise ValueError("Feature with no training observations")
            model = estimator(kind)
            model.fit(train[columns], train[target])
            estimate = model.predict(test[columns])
            scores.append({"protocol": protocol, "target": target, "model": name,
                "train_windows": len(train), "test_windows": len(test),
                "MAE_days": float(mean_absolute_error(y, estimate)),
                "RMSE_days": float(np.sqrt(mean_squared_error(y, estimate))),
                "R2": float(r2_score(y, estimate)), "negative_predictions": int((estimate < 0).sum()),
                "predictors": ";".join(columns)})
            for i, (_, row) in enumerate(test.iterrows()):
                predictions.append({"protocol": protocol, "target": target, "model": name,
                    "experiment_id": row.experiment_id, "assessment_utc": str(row.assessment_utc),
                    "true_days": float(y[i]), "predicted_days": float(estimate[i])})
    return scores, predictions


def run(source, out):
    out.mkdir(parents=True, exist_ok=False)
    with zipfile.ZipFile(source) as z:
        first, second = prepare(z, "FS369"), prepare(z, "FS437")
    train, test = first[first.eligible_checked].copy(), second[second.eligible_checked].copy()
    if train.assessment_utc.max() >= test.window_start.min():
        raise ValueError("Unexpected source/test overlap")
    boundary = int(len(train) * .8)
    late = train.iloc[boundary:].copy()
    early = train[train.assessment_utc.le(late.window_start.min() - pd.Timedelta(hours=1))].copy()
    protocols = [
        ("FS369_to_FS437_all_hours", train, test, True),
        ("FS369_to_FS437_no_logged_stim_hours", train[train.export_stimulation_records.eq(0)], test[test.export_stimulation_records.eq(0)], False),
        ("FS369_chronological_holdout", early, late, False),
    ]
    scores, predictions, splits = [], [], []
    for label, a, b, ablate in protocols:
        print(label, "train:", len(a), "test:", len(b), flush=True)
        s, p = evaluate(a, b, label, ablate)
        scores.extend(s)
        predictions.extend(p)
        for role, frame in [("train", a), ("test", b)]:
            splits.extend({"protocol": label, "role": role, "experiment_id": row.experiment_id,
                           "window_start": str(row.window_start), "assessment_utc": str(row.assessment_utc)}
                          for _, row in frame.iterrows())
    score, pred = pd.DataFrame(scores), pd.DataFrame(predictions)
    score.to_csv(out / "hourly_model_comparison.csv", index=False)
    pred.to_csv(out / "hourly_predictions.csv", index=False)
    pd.DataFrame(splits).to_csv(out / "split_membership.csv", index=False)
    combined = pd.concat([first, second], ignore_index=True)
    combined.to_csv(out / "audited_model_inputs.csv", index=False)
    drift = []
    for col in NEURAL + ENV:
        lo, hi = train[col].min(), train[col].max()
        observed = test[col].dropna()
        drift.append({"feature": col, "train_min": lo, "train_max": hi,
                      "test_min": observed.min(), "test_max": observed.max(),
                      "test_fraction_outside_train_range": float(((observed < lo) | (observed > hi)).mean())})
    pd.DataFrame(drift).to_csv(out / "feature_range_shift.csv", index=False)
    ablations = score[score.model.str.startswith("ablation_")].copy()
    full = score[(score.protocol == "FS369_to_FS437_all_hours") & score.model.eq("rich_neural_ridge")].set_index("target").MAE_days
    ablations["MAE_increase_vs_full_neural_ridge_days"] = ablations.MAE_days - ablations.target.map(full)
    ablations.to_csv(out / "neural_feature_ablations.csv", index=False)
    primary = score[score.protocol.eq("FS369_to_FS437_all_hours") & ~score.model.str.startswith("ablation_")]
    for target in TARGETS:
        s = primary[primary.target.eq(target)]
        fig, ax = plt.subplots(figsize=(11, 5), constrained_layout=True)
        ax.barh(s.model, s.MAE_days, color="#216b74")
        ax.invert_yaxis()
        ax.set_xlabel("Mean absolute error (days), 115 FS437 hours")
        ax.set_title("One-hour features: " + target.replace("_", " "))
        ax.grid(axis="x", alpha=.15)
        fig.savefig(out / (target + "_MAE.png"), dpi=170)
        plt.close(fig)
    status = {"input_sha256": hashlib.sha256(source.read_bytes()).hexdigest(),
        "protocols": [{"name": label, "train_hours": len(a), "test_hours": len(b)} for label, a, b, _ in protocols],
        "biological_lifespan_prediction_validated": False, "coverage_verified": False,
        "limitations": LIMITS, "software": {"python": platform.python_version(), "numpy": np.__version__, "pandas": pd.__version__, "sklearn": sklearn.__version__}}
    (out / "hourly_analysis_status.json").write_text(json.dumps(status, indent=2), encoding="utf-8")
    (out / "RESULTS_AND_LIMITATIONS.txt").write_text(primary[["target", "model", "MAE_days", "R2"]].to_string(index=False)
        + "\n\n" + "\n".join(LIMITS), encoding="utf-8")
    bundle = out / "Hourly_model_results.zip"
    with zipfile.ZipFile(bundle, "w", zipfile.ZIP_DEFLATED) as z:
        for path in sorted(out.iterdir()):
            if path.is_file() and path != bundle:
                z.write(path, path.name)
    print(primary[["target", "model", "MAE_days", "R2"]].to_string(index=False))
    print("\nHOURLY MODELS COMPLETE\nSaved:", bundle)
    return score, pred, status


if __name__ == "__main__":
    p = argparse.ArgumentParser(description=__doc__)
    p.add_argument("--input", type=Path, default=Path("FinalSpark_hourly_summary.zip"))
    p.add_argument("--output", type=Path)
    args = p.parse_args()
    if not args.input.is_file():
        p.error("Input ZIP not found: " + str(args.input))
    out = args.output or Path("hourly_model_results_" + datetime.now(timezone.utc).strftime("%Y%m%dT%H%M%S%fZ"))
    run(args.input, out)
