"""Calculate explicit endpoint tolerance scores from saved Windows predictions."""
from pathlib import Path
import argparse
import csv
import io

REPO = Path(__file__).resolve().parents[2]
DEFAULT = REPO / 'BDAPP_hourly_models/hourly_model_results_20261027T063832860540Z/hourly_predictions.csv'

def summarize(path):
    with Path(path).open(encoding='utf-8-sig', newline='') as stream:
        rows = list(csv.DictReader(stream))
    output = []
    for model in ['training_median', 'rich_neural_forest', 'neural_environment_forest']:
        selected = [r for r in rows if r['protocol'] == 'FS369_to_FS437_all_hours'
                    and r['target'] == 'days_until_recorded_experiment_endpoint'
                    and r['model'] == model]
        if len(selected) != 115 or len({r['assessment_utc'] for r in selected}) != 115:
            raise ValueError('Expected 115 unique primary FS437 test hours for ' + model)
        errors = [abs(float(r['predicted_days']) - float(r['true_days'])) for r in selected]
        for tolerance in [1, 2, 3, 5]:
            within = sum(error <= tolerance for error in errors)
            output.append(dict(source_run='windows_20261027T063832860540Z', model=model,
                               tolerance_days=tolerance, within_tolerance=within,
                               test_hours=len(errors), percent_within_tolerance=100 * within / len(errors)))
    return output

if __name__ == '__main__':
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument('--predictions', type=Path, default=DEFAULT)
    parser.add_argument('--output', type=Path, help='Optional new CSV; existing files are not overwritten')
    args = parser.parse_args()
    rows = summarize(args.predictions)
    buffer = io.StringIO(newline='')
    writer = csv.DictWriter(buffer, fieldnames=list(rows[0]), lineterminator='\n')
    writer.writeheader(); writer.writerows(rows)
    if args.output:
        with args.output.open('x', encoding='utf-8', newline='') as stream:
            stream.write(buffer.getvalue())
    print(buffer.getvalue(), end='')
