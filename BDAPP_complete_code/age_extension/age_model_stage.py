# Maral Farhatnejad | Student ID: 23012856 | BDAPP
"""Predict days after organoid dissociation from published five-minute well features.

This is a separate culture-age benchmark, not intact-organoid age or lifespan
prediction and not a one-hour-recording model. See analysis_plan.json.
"""
from pathlib import Path
from datetime import datetime, timezone
import argparse
import hashlib
import json
import platform
import zipfile
import numpy as np
import pandas as pd
import sklearn
from sklearn.dummy import DummyRegressor
from sklearn.ensemble import RandomForestRegressor
from sklearn.impute import SimpleImputer
from sklearn.linear_model import Ridge
from sklearn.pipeline import make_pipeline
from sklearn.preprocessing import StandardScaler
import matplotlib
matplotlib.use('Agg')
import matplotlib.pyplot as plt

ROOT = Path(__file__).resolve().parent
FEATURES = ['Mean Firing Rate (Hz)', 'Number of Bursts',
            'Area Under Normalized Cross-Correlation', 'Number of Network Bursts',
            'NDmean', 'sigEdgesMean']
LOG = ['Mean Firing Rate (Hz)', 'Number of Bursts', 'Number of Network Bursts']
MODELS = ['training_median', 'firing_rate_ridge', 'activity_ridge', 'activity_forest']


def require(condition, message):
    if not bool(condition):
        raise ValueError(message)


def prepare():
    expected = json.loads((ROOT / 'provenance/input_sha256.json').read_text())
    for name, digest in expected.items():
        require(hashlib.sha256((ROOT / 'data' / name).read_bytes()).hexdigest() == digest,
                'Source snapshot changed: ' + name)
    data = pd.read_csv(ROOT / 'data/Well_averages_developmenttimecourse_withNAP.csv')
    meta = pd.read_csv(ROOT / 'data/metadata.csv')
    meta.cell_line = meta.cell_line.replace({'KOLF2.1J': 'KOLF'})
    data = data.merge(meta, on='recording_filename_well', how='left', suffixes=('', '_meta'),
                      validate='one_to_one', indicator=True)
    require(data._merge.eq('both').all(), 'Recording missing from source metadata')
    for column in ['cell_line', 'batch', 'age_DPD']:
        require(data[column].eq(data[column + '_meta']).all(), 'Metadata conflict: ' + column)
    require(not data.duplicated(['plate_number', 'well_id', 'age_DPD']).any(), 'Repeated well/timepoint')
    allowed = data.condition.isna() | data.condition.str.lower().eq('vehicle')
    require(allowed.all(), 'Non-control treatment in published development table')
    require(set(data.age_DPD) == {6, 15, 22, 29, 37}, 'Unexpected source timepoints')
    plate_from_id = data.recording_filename_well.str.extract(r'^AP(\d+)_')[0].astype(int)
    age_from_id = data.recording_filename_well.str.extract(r'plusDIV(\d+)_')[0].astype(int)
    require(plate_from_id.eq(data.plate_number).all(), 'Plate ID conflict')
    require(age_from_id.eq(data.age_DPD).all(), 'Filename/age conflict')
    data['well_key'] = data.plate_number.astype(str) + ':' + data.well_id
    data['batch_key'] = data.cell_line + ':' + data.batch.astype(str)
    plates = sorted(data.plate_number.unique())
    parent = {int(p): int(p) for p in plates}
    def find(p):
        while parent[p] != p:
            parent[p] = parent[parent[p]]
            p = parent[p]
        return p
    for _, group in data.groupby('batch_key'):
        ps = sorted(group.plate_number.unique())
        for plate in ps[1:]:
            parent[find(int(plate))] = find(int(ps[0]))
    data['validation_group'] = data.plate_number.map(lambda p: str(find(int(p))))
    require(data.groupby('batch_key').validation_group.nunique().eq(1).all(), 'Differentiation batch split')
    for column in FEATURES:
        data[column] = pd.to_numeric(data[column], errors='raise')
        require(np.isfinite(data[column].dropna()).all(), 'Infinite feature: ' + column)
    for column in LOG:
        require(data[column].dropna().ge(0).all(), 'Negative count/rate')
        data['log_' + column] = np.log1p(data[column])
    return data


def metrics(truth, estimate):
    errors = truth - estimate
    return {'MAE_days': float(np.abs(errors).mean()),
            'RMSE_days': float(np.sqrt(np.square(errors).mean())),
            'R2': float(1 - np.square(errors).sum() / np.square(truth - truth.mean()).sum())}


def estimator(name):
    columns = ['log_' + c if c in LOG else c for c in FEATURES]
    if name in ['training_median', 'firing_rate_ridge']:
        columns = ['log_Mean Firing Rate (Hz)']
    model = DummyRegressor(strategy='median') if name == 'training_median' else (
        RandomForestRegressor(n_estimators=200, max_depth=4, min_samples_leaf=5,
                              random_state=42, n_jobs=1) if name == 'activity_forest' else Ridge(alpha=10))
    return columns, make_pipeline(SimpleImputer(strategy='median', add_indicator=True), StandardScaler(), model)


def folds(data, protocol):
    if protocol == 'unseen_cell_line_excluding_linked_plates':
        for line in sorted(data.cell_line.unique()):
            test = data[data.cell_line.eq(line)]
            train = data[~data.validation_group.isin(test.validation_group)]
            yield str(line), train, test
    else:
        selected = data[data.condition.isna()] if protocol == 'untreated_only_group_holdout' else data
        for group in sorted(selected.validation_group.unique()):
            yield str(group), selected[~selected.validation_group.eq(group)], selected[selected.validation_group.eq(group)]


def run(output):
    data = prepare()
    output.mkdir(parents=True, exist_ok=False)
    plan = json.loads((ROOT / 'analysis_plan.json').read_text())
    scores, predictions, memberships = [], [], []
    protocols = ['plate_and_batch_group_holdout', 'unseen_cell_line_excluding_linked_plates',
                 'untreated_only_group_holdout']
    for protocol in protocols:
        for fold, train, test in folds(data, protocol):
            require(len(train) >= 20 and len(test) >= 5, 'Insufficient fold size')
            for key in ['well_key', 'batch_key', 'validation_group', 'plate_number']:
                require(not set(train[key]) & set(test[key]), 'Leakage across ' + key)
            if protocol == 'unseen_cell_line_excluding_linked_plates':
                require(not set(train.cell_line) & set(test.cell_line), 'Cell-line leakage')
            print(protocol, fold, 'train', len(train), 'test', len(test), flush=True)
            for role, frame in [('train', train), ('test', test)]:
                part = frame[['recording_filename_well', 'cell_line', 'batch_key', 'plate_number', 'well_key', 'age_DPD', 'validation_group']].copy()
                part['protocol'], part['fold'], part['role'] = protocol, fold, role
                memberships.append(part)
            for name in MODELS:
                columns, model = estimator(name)
                require(train[columns].notna().any().all(), 'All-missing training feature')
                model.fit(train[columns], train.age_DPD)
                estimate = model.predict(test[columns])
                require(np.isfinite(estimate).all(), 'Non-finite estimates')
                if name == 'training_median':
                    require(np.allclose(estimate, train.age_DPD.median()), 'Median benchmark mismatch')
                scores.append(dict(protocol=protocol, fold=fold, model=name,
                    train_recordings=len(train), test_recordings=len(test),
                    **metrics(test.age_DPD.to_numpy(), estimate), predictors=';'.join(columns)))
                part = test[['recording_filename_well', 'cell_line', 'batch_key', 'plate_number', 'well_key', 'age_DPD', 'validation_group']].copy()
                part['protocol'], part['fold'], part['model'] = protocol, fold, name
                part['predicted_age_DPD'] = estimate
                predictions.append(part)
    score, pred = pd.DataFrame(scores), pd.concat(predictions, ignore_index=True)
    pooled = []
    for (protocol, name), part in pred.groupby(['protocol', 'model'], sort=False):
        require(not part.recording_filename_well.duplicated().any(), 'Repeated evaluation recording')
        expected = data[data.condition.isna()] if protocol == 'untreated_only_group_holdout' else data
        require(set(part.recording_filename_well) == set(expected.recording_filename_well), 'Incomplete held-out prediction set')
        baseline = pred[pred.protocol.eq(protocol) & pred.model.eq('training_median')].set_index('recording_filename_well')
        baseline_mae = np.abs(baseline.age_DPD - baseline.predicted_age_DPD).mean()
        result = metrics(part.age_DPD.to_numpy(), part.predicted_age_DPD.to_numpy())
        pooled.append(dict(protocol=protocol, model=name, recordings=len(part), folds=part.fold.nunique(),
            **result, MAE_change_vs_training_median_days=result['MAE_days'] - float(baseline_mae),
            macro_fold_MAE_days=float(score.loc[score.protocol.eq(protocol) & score.model.eq(name), 'MAE_days'].mean())))
    pooled = pd.DataFrame(pooled)
    score.to_csv(output / 'fold_scores.csv', index=False)
    pooled.to_csv(output / 'pooled_scores.csv', index=False)
    pred.to_csv(output / 'held_out_predictions.csv', index=False)
    age_scores = []
    for (protocol, name, age), part in pred.groupby(['protocol', 'model', 'age_DPD']):
        errors = part.predicted_age_DPD - part.age_DPD
        age_scores.append(dict(protocol=protocol, model=name, age_DPD=int(age), recordings=len(part),
            MAE_days=float(errors.abs().mean()), mean_prediction_DPD=float(part.predicted_age_DPD.mean()),
            mean_signed_error_days=float(errors.mean())))
    pd.DataFrame(age_scores).to_csv(output / 'age_specific_errors.csv', index=False)
    pd.concat(memberships, ignore_index=True).to_csv(output / 'fold_membership.csv', index=False)
    data.to_csv(output / 'audited_inputs.csv', index=False)
    data.groupby(['cell_line', 'batch', 'plate_number', 'validation_group', 'age_DPD']).size().rename('recordings').to_csv(output / 'source_inventory.csv')
    # Independently reconcile the saved per-fold scores from saved prediction rows.
    for row in score.itertuples(index=False):
        sub = pred[pred.protocol.eq(row.protocol) & pred.fold.eq(row.fold) & pred.model.eq(row.model)]
        actual = metrics(sub.age_DPD.to_numpy(), sub.predicted_age_DPD.to_numpy())
        require(all(np.isclose(actual[c], getattr(row, c), rtol=1e-10) for c in actual), 'Saved score inconsistency')
    primary = pred[pred.protocol.eq('plate_and_batch_group_holdout')]
    fig, axes = plt.subplots(1, 3, figsize=(12, 4), sharex=True, sharey=True, constrained_layout=True)
    for ax, name in zip(axes, MODELS[1:]):
        sub = primary[primary.model.eq(name)]
        ax.scatter(sub.age_DPD, sub.predicted_age_DPD, s=10, alpha=.25, color='#216b74')
        ax.plot([6, 37], [6, 37], '--', color='black', linewidth=1)
        ax.set_title(name.replace('_', ' '))
        ax.set_xlabel('Actual days after dissociation')
        ax.grid(alpha=.15)
    axes[0].set_ylabel('Held-out predicted days after dissociation')
    fig.suptitle('Published five-minute recordings: whole plate/batch groups held out')
    fig.savefig(output / 'held_out_culture_age.png', dpi=170)
    plt.close(fig)
    fig, ax = plt.subplots(figsize=(9, 4), constrained_layout=True)
    subset = pooled[pooled.protocol.eq('plate_and_batch_group_holdout')]
    ax.barh(subset.model, subset.MAE_days, color='#216b74')
    ax.invert_yaxis()
    ax.set_xlabel('Pooled held-out mean absolute error (days after dissociation)')
    ax.set_title('All four models, eight plate/batch groups')
    fig.savefig(output / 'model_MAE.png', dpi=170)
    plt.close(fig)
    status = dict(target=plan['target'], source_doi=plan['source_doi'],
        recordings=len(data), wells=int(data.well_key.nunique()), cell_lines=int(data.cell_line.nunique()),
        differentiation_batches=int(data.batch_key.nunique()), plates=int(data.plate_number.nunique()),
        independent_validation_components=int(data.validation_group.nunique()),
        sampled_ages_DPD=sorted(map(int, data.age_DPD.unique())),
        split_and_metric_checks='PASS', score_rows=len(score), held_out_prediction_rows=len(pred),
        plan_sha256=hashlib.sha256((ROOT / 'analysis_plan.json').read_bytes()).hexdigest(),
        source_sha256=hashlib.sha256(Path(__file__).read_bytes()).hexdigest(),
        software=dict(python=platform.python_version(), numpy=np.__version__, pandas=pd.__version__, sklearn=sklearn.__version__),
        intact_organoid_age_prediction_validated=False, lifespan_prediction_performed=False,
        limits=plan['limits'])
    (output / 'analysis_status.json').write_text(json.dumps(status, indent=2), encoding='utf-8')
    (output / 'RESULTS.txt').write_text('Separate culture-age benchmark: days after organoid dissociation\n'
        + 'Published five-minute well averages; not intact-organoid age, lifespan or a one-hour recording model.\n\n'
        + pooled.to_string(index=False) + '\n\n' + '\n'.join(plan['limits']), encoding='utf-8')
    bundle = output / 'Age_extension_results.zip'
    with zipfile.ZipFile(bundle, 'w', zipfile.ZIP_DEFLATED) as archive:
        for file in sorted(output.iterdir()):
            if file.is_file() and file != bundle:
                archive.write(file, file.name)
    print('\nAGE BENCHMARK COMPLETE — SPLIT AND METRIC CHECKS PASSED')
    print(pooled.to_string(index=False))
    print('Saved:', bundle)
    return pooled, status


if __name__ == '__main__':
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument('--output', type=Path)
    args = parser.parse_args()
    run(args.output or ROOT / 'results' / datetime.now(timezone.utc).strftime('%Y%m%dT%H%M%S%fZ'))
