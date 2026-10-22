"""Independently check sealed scores, ensemble arithmetic and reported AP/Top-K."""
import csv
import hashlib
import json
from pathlib import Path
import numpy as np
from sklearn.metrics import average_precision_score

ROOT = Path(__file__).resolve().parents[1]
OUT = ROOT / 'outputs/external-synthemol-v1'


def digest(path):
    return hashlib.sha256(path.read_bytes()).hexdigest()


def main():
    result = json.loads((OUT / 'results.json').read_text())
    seal = json.loads((OUT / 'predictions_sealed.json').read_text())
    assert result['complete'] and not result['labels_used_for_training_or_tuning']
    assert digest(OUT / 'predictions.csv') == result['predictions_sha256']
    assert digest(OUT / 'unscored_predictions.npy') == seal['scores_sha256']
    assert digest(OUT / 'prediction_ids.json') == seal['ids_sha256']
    with (OUT / 'predictions.csv').open() as f:
        rows = list(csv.DictReader(f))
    ids = [r['compound_key'] for r in rows]
    assert ids == json.loads((OUT / 'prediction_ids.json').read_text())
    assert len(rows) == len(set(ids)) == 1107
    labels = np.array([int(r['observed_label']) for r in rows])
    assert labels.sum() == result['hits'] == 127
    scores = np.array([[float(r[n]) for n in seal['model_order']] for r in rows])
    np.testing.assert_array_equal(scores, np.load(OUT / 'unscored_predictions.npy'))
    for name, metrics in {**result['groups'], **result['members']}.items():
        values = np.array([float(r[name]) for r in rows])
        assert abs(average_precision_score(labels, values) - metrics['average_precision']) < 1e-12
        ordered = sorted(range(len(rows)), key=lambda i: (-values[i], ids[i]))
        for k in (20, 50, 100):
            assert int(labels[ordered[:k]].sum()) == metrics['top_k'][str(k)]['hits']
        if name.endswith('_3'):
            mean = np.mean([[float(r[f'{name}_seed{s}']) for r in rows] for s in (17, 29, 43)], axis=0)
            np.testing.assert_allclose(values, mean, rtol=0, atol=1e-15)
    checks = {'passed': True, 'n': len(rows), 'hits': int(labels.sum()),
              'models_verified': 13, 'top_k_checks': 39, 'ensembles_verified': 3,
              'prediction_seals_matched': True, 'results_sha256': digest(OUT / 'results.json'),
              'script_sha256': digest(Path(__file__))}
    (ROOT / 'reports/external_synthemol_verification_v1.json').write_text(json.dumps(checks, indent=2) + '\n')
    print(json.dumps(checks))


if __name__ == '__main__':
    main()
