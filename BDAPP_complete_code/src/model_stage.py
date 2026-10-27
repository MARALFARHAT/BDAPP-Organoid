# Maral Farhatnejad | Student ID: 23012856 | BDAPP
"""Exploratory six-hour baselines from BDAPP_model_inputs.zip.

Run locally:
  python model_stage.py --input F:/BDAPP-Organoid/outputs/BDAPP_model_inputs.zip

Targets describe an aggregate experiment. They do not establish an individual
organoid's biological health, independently verified death, or remaining life.
Only explicitly whitelisted activity/environment features enter neural models.
"""
from __future__ import annotations
import argparse
from datetime import datetime, timezone
import hashlib
import io
import json
from pathlib import Path
import platform
import zipfile

import numpy as np
import pandas as pd
import sklearn
from sklearn.dummy import DummyRegressor
from sklearn.ensemble import RandomForestRegressor
from sklearn.impute import SimpleImputer
from sklearn.linear_model import Ridge
from sklearn.metrics import mean_absolute_error, mean_squared_error, r2_score
from sklearn.pipeline import make_pipeline
from sklearn.preprocessing import StandardScaler
import matplotlib
matplotlib.use("Agg")
import matplotlib.pyplot as plt

TARGETS = ["batch_calendar_age_days", "days_until_recorded_experiment_endpoint"]
ENV = ["temperature_mean", "co2_mean", "o2_mean"]
SPECIFICATIONS = {
    "training_median": (["log_event_count"], "median", "baseline"),
    "event_count_ridge": (["log_event_count"], "ridge", "activity"),
    "environment_ridge": (ENV, "ridge", "environment"),
    "event_environment_ridge": (["log_event_count"] + ENV, "ridge", "combined"),
    "event_environment_stimulation_ridge": (["log_event_count", "log_stimulation_count"] + ENV, "ridge", "combined"),
    "event_environment_forest": (["log_event_count"] + ENV, "forest", "combined"),
    "recording_clock_diagnostic": (["elapsed_recording_days"], "ridge_clock", "NON_NEURAL_DIAGNOSTIC"),
}
LIMITATIONS = [
    "Only two aggregate experiment records, both labeled NS Batch 3 with the same Batch Start. Biological independence is unverified.",
    "Number of NS is 12 in both records. No electrode-to-neurosphere mapping or individual endpoints were supplied.",
    "Batch age is arithmetic from the supplied Batch Start. It is not an independently measured maturation state.",
    "Recorded endpoints have activity-based causes ('Almost no activity' / 'No activity'); the endpoint task may be circular and lacks independently verified death labels.",
    "Six-hour counts are not exposure-normalized rates. Partial observation, acquisition settings and stimulation artifacts may drive patterns.",
    "An absent stimulation aggregate remains unknown. Training-median imputation and missingness indicators may encode acquisition differences.",
    "Environmental units and causal effects are not verified. Reported values enter exploratory models only.",
    "FS369 and FS437 cover non-overlapping age ranges. Transfer tests require substantial extrapolation.",
    "Temporal windows are dependent. No window-level confidence intervals or population-generalization claims are reported.",
    "Hyperparameters are fixed. All compared models and negative results are retained. FS437 is now an inspected evaluation set, not untouched future confirmation.",
    "These six-hour baseline inputs cannot implement the original one-hour timing/burst/synchrony feature plan.",
    "DANDI and Zenodo data are absent from this ZIP. No external validation has run.",
    "Both supplied README copies describe FS369. FS437 acquisition conventions need their own confirmation.",
    "Within each experiment, batch age and time to the recorded endpoint sum to a constant. Their within-experiment errors are not independent confirmations.",
    "A recording-clock diagnostic can reconstruct time-based targets within one experiment. Its near-zero internal error is arithmetic, not neural age prediction.",
]


def prepare(archive, experiment):
    df = pd.read_csv(io.BytesIO(archive.read(f"{experiment}/merged_window_features.csv")))
    metadata = json.loads(archive.read(f"{experiment}/metadata.json"))
    if len(metadata) != 1:
        raise ValueError(f"{experiment}: expected one aggregate metadata record")
    meta = metadata[0]
    birth = pd.Timestamp(meta["Batch Start"], tz="UTC")
    recording_start = pd.Timestamp(meta["Date"], tz="UTC")
    endpoint = pd.Timestamp(meta["Dead Date"], tz="UTC")
    if not birth <= recording_start < endpoint:
        raise ValueError(f"{experiment}: inconsistent metadata dates")
    df["window_start"] = pd.to_datetime(df["window_start"], utc=True, errors="raise")
    if df.window_start.duplicated().any():
        raise ValueError(f"{experiment}: duplicate window timestamps")
    df = df.sort_values("window_start").reset_index(drop=True)
    for col in ["detected_events", "stimulation_records"] + ENV:
        df[col] = pd.to_numeric(df[col], errors="raise")
        if not np.isfinite(df[col].dropna()).all():
            raise ValueError(f"{experiment}: non-finite {col}")
    for col in ["detected_events", "stimulation_records"]:
        if df[col].dropna().lt(0).any():
            raise ValueError(f"{experiment}: negative record counts")
    # The whole feature window precedes the target assessment at its end.
    # Coverage is still unknown; this is not evidence of six observed hours.
    df["assessment_utc"] = df.window_start + pd.Timedelta(hours=6)
    df["experiment_id"] = experiment
    df["batch_calendar_age_days"] = (df.assessment_utc - birth).dt.total_seconds() / 86400
    df["days_until_recorded_experiment_endpoint"] = (endpoint - df.assessment_utc).dt.total_seconds() / 86400
    df["elapsed_recording_days"] = (df.assessment_utc - recording_start).dt.total_seconds() / 86400
    df["log_event_count"] = np.log1p(df.detected_events)
    df["log_stimulation_count"] = np.log1p(df.stimulation_records)
    df["exclusion_reason"] = ""
    df.loc[df.window_start.lt(recording_start), "exclusion_reason"] = "window starts before metadata recording date"
    df.loc[df.detected_events.isna(), "exclusion_reason"] = "no event count; exposure/silence unknown"
    df.loc[df.assessment_utc.gt(endpoint), "exclusion_reason"] = "window extends beyond recorded endpoint"
    df["eligible"] = df.exclusion_reason.eq("")
    summary = {
        "experiment_id": experiment, "batch_name": meta.get("Batch Name"),
        "batch_start_utc": str(birth), "recording_start_utc": str(recording_start),
        "recorded_endpoint_utc": str(endpoint), "recorded_endpoint_cause": meta.get("Dead Cause"),
        "reported_number_of_neurospheres": meta.get("Number of NS"),
        "total_windows": len(df), "eligible_windows": int(df.eligible.sum()),
        "batch_age_min_days": float(df.loc[df.eligible, TARGETS[0]].min()),
        "batch_age_max_days": float(df.loc[df.eligible, TARGETS[0]].max()),
        "detected_events_aggregate_total": int(df.detected_events.sum()),
    }
    return df, summary


def make_model(kind):
    if kind == "median":
        return make_pipeline(SimpleImputer(strategy="median"), DummyRegressor(strategy="median"))
    if kind == "forest":
        return make_pipeline(SimpleImputer(strategy="median", add_indicator=True),
                             RandomForestRegressor(n_estimators=200, max_depth=4,
                                                   min_samples_leaf=5, random_state=42, n_jobs=1))
    return make_pipeline(SimpleImputer(strategy="median", add_indicator=True), StandardScaler(),
                         Ridge(alpha=1e-8 if kind == "ridge_clock" else 10.0))


def evaluate(protocol, training, testing):
    scores, predictions = [], []
    if len(training) < 20 or len(testing) < 5:
        raise ValueError("Insufficient eligible windows for the specified exploratory test")
    for target in TARGETS:
        for name, (features, kind, family) in SPECIFICATIONS.items():
            if any(training[c].notna().sum() == 0 for c in features):
                raise ValueError(f"No training observations for {name}")
            model = make_model(kind)
            # Fitting the pipeline only here prevents test-set scaling/imputation.
            model.fit(training[features], training[target])
            estimate = model.predict(testing[features])
            truth = testing[target].to_numpy()
            baseline_error = float(np.mean(np.abs(truth - np.median(training[target]))))
            mae = float(mean_absolute_error(truth, estimate))
            scores.append({
                "protocol": protocol, "target": target, "model": name, "family": family,
                "train_windows": len(training), "test_windows": len(testing),
                "MAE_days": mae, "RMSE_days": float(np.sqrt(mean_squared_error(truth, estimate))),
                "R2": float(r2_score(truth, estimate)),
                "MAE_change_vs_training_median_days": mae - baseline_error,
                "negative_predictions": int((estimate < 0).sum()), "features": ";".join(features),
            })
            for position, (_, row) in enumerate(testing.iterrows()):
                predictions.append({"protocol": protocol, "target": target, "model": name,
                                    "experiment_id": row.experiment_id, "window_start": str(row.window_start),
                                    "assessment_utc": str(row.assessment_utc), "true_days": float(truth[position]),
                                    "predicted_days": float(estimate[position])})
    return scores, predictions


def write_figures(scores, predictions, all_data, out):
    colors = ["#596475", "#247b86", "#7b638f", "#14545a", "#b07936", "#4b8071", "#a45757"]
    for target in TARGETS:
        data = scores[(scores.protocol == "FS369_to_FS437") & scores.target.eq(target)]
        fig, ax = plt.subplots(figsize=(10, 5), constrained_layout=True)
        ax.barh(data.model, data.MAE_days, color=colors)
        ax.invert_yaxis()
        ax.set_xlabel("Mean absolute error (days), 20 FS437 windows")
        ax.set_title("FS369 to FS437: " + target.replace("_", " "))
        ax.grid(axis="x", alpha=.18)
        fig.savefig(out / f"transfer_MAE_{target}.png", dpi=180)
        plt.close(fig)
    fig, axes = plt.subplots(2, 1, figsize=(11, 8), constrained_layout=True)
    for ax, target in zip(axes, TARGETS):
        p = predictions[(predictions.protocol == "FS369_to_FS437") & predictions.target.eq(target)]
        for name in ["training_median", "event_count_ridge", "event_environment_ridge"]:
            d = p[p.model.eq(name)]
            ax.plot(pd.to_datetime(d.assessment_utc, utc=True), d.predicted_days, "o-", label=name, markersize=3)
        d = p[p.model.eq("training_median")]
        ax.plot(pd.to_datetime(d.assessment_utc, utc=True), d.true_days, color="black", label="metadata-derived target", linewidth=2)
        ax.set_title(target.replace("_", " "))
        ax.set_ylabel("Days")
        ax.legend(fontsize=8)
        ax.grid(alpha=.18)
    axes[-1].set_xlabel("FS437 assessment time (UTC)")
    fig.savefig(out / "transfer_predictions.png", dpi=180)
    plt.close(fig)
    fig, axes = plt.subplots(1, 2, figsize=(11, 4), constrained_layout=True)
    for experiment, group in all_data[all_data.eligible].groupby("experiment_id"):
        axes[0].scatter(group[TARGETS[0]], group.detected_events, s=15, label=experiment)
        axes[1].scatter(group[TARGETS[0]], group.temperature_mean, s=15, label=experiment)
    axes[0].set_ylabel("Detected events per clock window")
    axes[1].set_ylabel("Temperature (reported source value)")
    for ax in axes:
        ax.set_xlabel("Batch calendar age (days)")
        ax.grid(alpha=.18)
        ax.legend()
    fig.savefig(out / "age_and_environment_shift.png", dpi=180)
    plt.close(fig)


def run(input_zip, out):
    out.mkdir(parents=True, exist_ok=False)
    with zipfile.ZipFile(input_zip) as archive:
        first, first_summary = prepare(archive, "FS369")
        second, second_summary = prepare(archive, "FS437")
        electrode = pd.read_csv(io.BytesIO(archive.read("FS369/event_electrode_counts.csv")))
        if int(electrode.detected_events.sum()) != int(first.detected_events.sum()):
            raise ValueError("FS369 per-electrode totals disagree with aggregate totals")
    training = first[first.eligible].copy()
    testing = second[second.eligible].copy()
    boundary = int(np.floor(len(training) * .8))
    if training.assessment_utc.max() >= testing.window_start.min():
        raise ValueError("Unexpected source/test temporal overlap")
    # One full clock window embargo, even when eligible rows contain gaps.
    test_late = training.iloc[boundary:].copy()
    cutoff = test_late.window_start.min() - pd.Timedelta(hours=6)
    train_early = training.loc[training.assessment_utc.le(cutoff)].copy()
    if train_early.assessment_utc.max() > test_late.window_start.min() - pd.Timedelta(hours=6):
        raise AssertionError("Temporal embargo failed")
    scores, predictions = evaluate("FS369_to_FS437", training, testing)
    more_scores, more_predictions = evaluate("FS369_chronological_holdout", train_early, test_late)
    scores = pd.DataFrame(scores + more_scores)
    predictions = pd.DataFrame(predictions + more_predictions)
    all_data = pd.concat([first, second], ignore_index=True)
    scores.to_csv(out / "model_comparison.csv", index=False)
    predictions.to_csv(out / "predictions.csv", index=False)
    all_data.to_csv(out / "labeled_windows_and_exclusions.csv", index=False)
    pd.DataFrame([first_summary, second_summary]).to_csv(out / "experiment_summary.csv", index=False)
    status = {
        "stage": "exploratory six-hour aggregate baselines",
        "input_sha256": hashlib.sha256(input_zip.read_bytes()).hexdigest(),
        "protocols": {"FS369_to_FS437": {"train": len(training), "test": len(testing)},
                      "FS369_chronological_holdout": {"train": len(train_early), "test": len(test_late), "embargo_hours": 6}},
        "individual_organoid_lifespan_model": "not established; mapping and independent endpoints unavailable",
        "original_one_hour_spontaneous_feature_model": "not run; requires event-level extraction and coverage verification",
        "external_validation": "not run; DANDI/Zenodo files not included",
        "population_uncertainty": "not estimated from dependent windows",
        "software": {"python": platform.python_version(), "pandas": pd.__version__,
                     "numpy": np.__version__, "sklearn": sklearn.__version__},
        "limitations": LIMITATIONS,
    }
    (out / "analysis_status.json").write_text(json.dumps(status, indent=2), encoding="utf-8")
    write_figures(scores, predictions, all_data, out)
    transfer = scores[scores.protocol.eq("FS369_to_FS437")]
    lines = ["BDAPP: actual six-hour baseline results", "", "Train FS369, test FS437. No random mixing of experiment windows.",
             "Age label: assessment timestamp minus documented Batch Start.",
             "Endpoint label: metadata Dead Date minus assessment timestamp.",
             "Both are aggregate experiment labels. The endpoint task is not validated individual survival prediction.", ""]
    for target in TARGETS:
        lines.extend([target, transfer[transfer.target.eq(target)][["model", "MAE_days", "R2"]].to_string(index=False), ""])
    lines.extend(["Limitations"] + ["- " + x for x in LIMITATIONS])
    lines.extend(["", "Reproduction", "python model_stage.py --input BDAPP_model_inputs.zip",
                  "Dependencies: numpy pandas matplotlib scikit-learn", "",
                  "Methods reference: https://scikit-learn.org/stable/common_pitfalls.html",
                  "Labels and experimental context: README.md and metadata.json within the supplied ZIP."])
    (out / "RESULTS_AND_LIMITATIONS.txt").write_text("\n".join(lines), encoding="utf-8")
    print(transfer[["target", "model", "MAE_days", "R2"]].to_string(index=False))
    print("\nSaved:", out)
    return scores, predictions, status


if __name__ == "__main__":
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--input", type=Path, default=Path("BDAPP_model_inputs.zip"))
    parser.add_argument("--output", type=Path)
    args = parser.parse_args()
    if not args.input.is_file():
        parser.error(f"Input ZIP not found: {args.input}")
    output = args.output or args.input.parent / ("model_results_" + datetime.now(timezone.utc).strftime("%Y%m%dT%H%M%S%fZ"))
    run(args.input, output)
