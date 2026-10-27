# Maral Farhatnejad | Student ID: 23012856 | BDAPP
from pathlib import Path
import hashlib
import io
import json
import zipfile
import numpy as np
import pandas as pd
import hourly_models
import model_stage


def require(condition, message):
    if not bool(condition):
        raise ValueError(message)


def read_csv(archive, name):
    return pd.read_csv(io.BytesIO(archive.read(name)))


def check_inputs(project):
    manifest = json.loads((project / 'provenance/input_manifest.json').read_text())
    for record in manifest:
        path = project / record['path']
        require(path.stat().st_size == record['bytes'], 'Input size changed: ' + record['path'])
        require(hashlib.sha256(path.read_bytes()).hexdigest() == record['sha256'],
                'Input hash changed: ' + record['path'])
    inventory, frames = [], {}
    with zipfile.ZipFile(project / 'data/FinalSpark_hourly_summary.zip') as archive:
        for experiment in ['FS369', 'FS437']:
            frame = hourly_models.prepare(archive, experiment)
            audit = json.loads(archive.read(experiment + '/extraction_audit.json'))
            observed = frame.detected_events.notna()
            channels = frame.loc[observed, 'active_observed_channels']
            require((frame.loc[observed, 'channel_interval_count'] ==
                     frame.loc[observed, 'detected_events'] - channels).all(),
                    experiment + ': channel/interval bookkeeping mismatch')
            require(frame.verified_recording_seconds.isna().all() and
                    frame.exposure_normalized_rate_hz.isna().all(), 'Unverified exposure has been filled')
            require(frame.loc[~observed, 'detected_events'].isna().all(), 'Missing event windows changed')
            require(int(observed.sum()) == audit['event_hours'], 'Event-hour audit mismatch')
            require(len(frame) == audit['clock_hours'], 'Clock-hour audit mismatch')
            require(int(frame.eligible_checked.sum()) == audit['eligible_hours'], 'Eligibility audit mismatch')
            for column in ['fraction_events_in_channel_bursts', 'largest_channel_event_share',
                           'fraction_channel_intervals_under_10ms']:
                require(frame[column].dropna().between(0, 1).all(), 'Invalid fraction: ' + column)
            frames[experiment] = frame
            inventory.append(dict(dataset=experiment, track='exploratory experiment-level models',
                recordings_or_grid_windows=len(frame), event_containing_windows=int(observed.sum()),
                eligible_model_windows=int(frame.eligible_checked.sum()),
                detected_events=int(frame.detected_events.sum()), units='aggregate experiment'))
    with zipfile.ZipFile(project / 'data/BDAPP_model_inputs.zip') as archive:
        for experiment in ['FS369', 'FS437']:
            frame, _ = model_stage.prepare(archive, experiment)
            require(int(frame.detected_events.sum()) == int(frames[experiment].detected_events.sum()),
                    'Six-hour and one-hour source event totals disagree: ' + experiment)
            frames['six_' + experiment] = frame
        channels = read_csv(archive, 'FS369/event_electrode_counts.csv')
        require(int(channels.detected_events.sum()) == int(frames['FS369'].detected_events.sum()),
                'Six-hour electrode totals disagree')
    with zipfile.ZipFile(project / 'data/DANDI_summary.zip') as archive:
        data = read_csv(archive, 'dandi_recording_features.csv')
        status = json.loads(archive.read('dandi_analysis_status.json'))
        audit = read_csv(archive, 'dandi_file_audit.csv')
        cultures = read_csv(archive, 'dandi_culture_summary.csv')
        require(len(data) == status['processed_files'] == int(audit.status.eq('processed').sum()),
                'DANDI processed-count mismatch')
        require(status['skipped_files'] == 0, 'Packaged DANDI run has skipped files')
        require(not data.recording_id.duplicated().any(), 'DANDI recording repeated')
        require(not data.duplicated(['culture_id', 'session_start_utc']).any(), 'DANDI session repeated')
        require(data.culture_age_days.isna().all(), 'Unknown DANDI culture ages have been filled')
        require((data.channel_isi_count == data.detected_events - data.active_source_channels).all(),
                'DANDI channel/interval bookkeeping mismatch')
        require(np.allclose(data.events_per_nominal_recording_second,
                            data.detected_events / data.nominal_duration_s), 'DANDI nominal rate mismatch')
        require(int(cultures.detected_events.sum()) == int(data.detected_events.sum()),
                'DANDI culture/event totals disagree')
        dates = pd.to_datetime(data.session_start_utc, utc=True)
        elapsed = (dates - dates.groupby(data.culture_id).transform('min')).dt.total_seconds() / 86400
        require(np.allclose(elapsed, data.recording_elapsed_days), 'DANDI elapsed-time mismatch')
        require(data.culture_id.nunique() == status['cultures'], 'DANDI culture-count mismatch')
        inventory.append(dict(dataset='DANDI 002020', track='descriptive rat cortical cultures',
            recordings_or_grid_windows=len(data), event_containing_windows=None, eligible_model_windows=None,
            detected_events=int(data.detected_events.sum()), units=str(data.culture_id.nunique()) + ' cultures'))
    with zipfile.ZipFile(project / 'data/Zenodo_summary.zip') as archive:
        data = read_csv(archive, 'zenodo_organoid_features.csv')
        status = json.loads(archive.read('zenodo_analysis_status.json'))
        audit = read_csv(archive, 'zenodo_file_audit.csv')
        require(len(data) == status['processed_organoids'] == int(audit.status.eq('processed').sum()),
                'Zenodo processed-count mismatch')
        require(not audit.status.isin(['skipped', 'CONFLICT_DIFFERENT_CONTENT']).any(),
                'Zenodo skipped/conflicting files')
        require(not data.organoid_id.duplicated().any(), 'Zenodo organoid repeated')
        require(data.biological_age_days.isna().all() and data.remaining_life_days.isna().all(),
                'Unknown Zenodo labels have been filled')
        require(np.allclose(data.events_per_declared_second, data.detected_events / data.declared_duration_s),
                'Zenodo declared-duration rate mismatch')
        require(np.allclose(data.mean_events_per_declared_channel_second,
                            data.detected_events / data.declared_duration_s / data.declared_channels),
                'Zenodo declared-channel rate mismatch')
        require(int(data.detected_events.sum()) == status['detected_events'], 'Zenodo total mismatch')
        require(data.experiment_prefix.nunique() == status['experiment_prefixes'], 'Zenodo prefix mismatch')
        affected = data[data.amplitude_mismatch_channel_count.gt(0)]
        require(affected.organoid_id.tolist() == ['fs417-org0'], 'Unexpected Zenodo amplitude audit')
        require(int(affected.detected_events.sum()) == 33096 and int(affected.valid_amplitude_events.sum()) == 0,
                'Zenodo valid times or withheld amplitudes changed')
        inventory.append(dict(dataset='Zenodo 22755004', track='descriptive human forebrain organoids',
            recordings_or_grid_windows=len(data), event_containing_windows=None, eligible_model_windows=None,
            detected_events=int(data.detected_events.sum()), units=str(data.experiment_prefix.nunique()) + ' experiment prefixes'))
    return inventory, frames


def check_predictions(scores, predictions, frames, hourly):
    keys = ['protocol', 'target', 'model']
    require(not scores.duplicated(keys).any(), 'Repeated score row')
    require(not predictions.duplicated(keys + ['experiment_id', 'assessment_utc']).any(), 'Repeated prediction')
    require(len(predictions.groupby(keys)) == len(scores), 'Score/prediction model groups disagree')
    for row in scores.itertuples(index=False):
        part = predictions[(predictions.protocol == row.protocol) &
                           (predictions.target == row.target) & (predictions.model == row.model)]
        require(len(part) == row.test_windows, 'Prediction/test count mismatch')
        y, estimate = part.true_days.to_numpy(), part.predicted_days.to_numpy()
        require(np.isfinite(y).all() and np.isfinite(estimate).all(), 'Non-finite prediction')
        metrics = [np.abs(y - estimate).mean(), np.sqrt(np.square(y - estimate).mean()),
                   1 - np.square(y - estimate).sum() / np.square(y - y.mean()).sum()]
        require(np.allclose(metrics, [row.MAE_days, row.RMSE_days, row.R2], rtol=1e-9, atol=1e-9),
                'Metric calculation mismatch: ' + row.model)
        require(int((estimate < 0).sum()) == row.negative_predictions, 'Negative estimates not retained')
        experiment = 'FS369' if row.protocol == 'FS369_chronological_holdout' else 'FS437'
        require(part.experiment_id.eq(experiment).all(), 'Unexpected evaluation experiment')
        frame = frames[experiment if hourly else 'six_' + experiment]
        lookup = frame.set_index('assessment_utc')[row.target]
        times = pd.to_datetime(part.assessment_utc, utc=True)
        require(np.allclose(y, lookup.loc[times].to_numpy()), 'Prediction truth differs from metadata')
        if hourly:
            if row.model.startswith('ablation_without_'):
                removed = hourly_models.GROUPS[row.model.removeprefix('ablation_without_')]
                allowed = [c for c in hourly_models.NEURAL if c not in removed]
            else:
                allowed = hourly_models.SPECS[row.model][0]
            require(row.predictors.split(';') == allowed, 'Unexpected predictor/possible target leakage')
        else:
            require(row.features.split(';') == model_stage.SPECIFICATIONS[row.model][0], 'Unexpected baseline predictors')
    return len(scores), len(predictions)


def check_results(project, result, inventory=None, frames=None):
    if inventory is None or frames is None:
        inventory, frames = check_inputs(project)
    hourly = pd.read_csv(result / 'hourly_models/hourly_model_comparison.csv')
    hp = pd.read_csv(result / 'hourly_models/hourly_predictions.csv')
    baseline = pd.read_csv(result / 'six_hour_baselines/model_comparison.csv')
    bp = pd.read_csv(result / 'six_hour_baselines/predictions.csv')
    hn, hpn = check_predictions(hourly, hp, frames, True)
    bn, bpn = check_predictions(baseline, bp, frames, False)
    require(hn == 58 and bn == 28, 'Missing prespecified comparisons or ablations')
    splits = pd.read_csv(result / 'hourly_models/split_membership.csv')
    for protocol, group in splits.groupby('protocol'):
        train, test = group[group.role.eq('train')], group[group.role.eq('test')]
        train_keys = set(zip(train.experiment_id, train.window_start))
        test_keys = set(zip(test.experiment_id, test.window_start))
        require(not train_keys & test_keys, 'Train/test window overlap')
        require(train.experiment_id.eq('FS369').all(), 'Training includes transfer test experiment')
        if protocol == 'FS369_chronological_holdout':
            require(test.experiment_id.eq('FS369').all(), 'Unexpected chronology test experiment')
            require(pd.to_datetime(train.assessment_utc, utc=True).max() <=
                    pd.to_datetime(test.window_start, utc=True).min() - pd.Timedelta(hours=1),
                    'One-hour chronology embargo failed')
        else:
            require(test.experiment_id.eq('FS437').all(), 'Transfer test includes training experiment')
        for role, part in [('train', train), ('test', test)]:
            expected = hourly.loc[hourly.protocol.eq(protocol), role + '_windows']
            require(expected.eq(len(part)).all(), 'Split size differs from model scores')
            for experiment, sub in part.groupby('experiment_id'):
                frame = frames[experiment].set_index('window_start')
                rows = frame.loc[pd.to_datetime(sub.window_start, utc=True)]
                require(rows.eligible_checked.all(), 'Ineligible model window')
                if 'no_logged_stim' in protocol:
                    require(rows.export_stimulation_records.eq(0).all(), 'Logged-stimulation sensitivity split wrong')
    source = frames['six_FS369']
    eligible = source[source.eligible]
    late = eligible.iloc[int(np.floor(len(eligible) * .8)):]
    early = eligible[eligible.assessment_utc.le(late.window_start.min() - pd.Timedelta(hours=6))]
    chrono = baseline[baseline.protocol.eq('FS369_chronological_holdout')]
    require(chrono.train_windows.eq(len(early)).all() and chrono.test_windows.eq(len(late)).all(),
            'Six-hour chronology/embargo size mismatch')
    primary = hourly[hourly.protocol.eq('FS369_to_FS437_all_hours') & ~hourly.model.str.startswith('ablation_')]
    return dict(status='PASS', input_archives_hash_verified=4,
        scope='Derived-table bookkeeping, metadata labels, predictor lists, splits and prediction metrics; raw acquisition and biological endpoints remain unverified.',
        inventory=inventory, hourly_score_rows=hn, hourly_prediction_rows=hpn,
        six_hour_score_rows=bn, six_hour_prediction_rows=bpn,
        all_primary_hourly_transfer_R2_negative=bool(primary.R2.lt(0).all()),
        zenodo_amplitude_mismatch_recording='fs417-org0',
        biological_lifespan_prediction_validated=False, continuous_recording_exposure_verified=False)
