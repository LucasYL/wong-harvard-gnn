"""Independently recalculate RF metrics from saved scores without fitting models."""
import csv
import json
from pathlib import Path

import numpy as np
from sklearn.metrics import average_precision_score, roc_auc_score

from antibacterial.common import sha256

ROOT = Path(__file__).resolve().parents[1]


def main():
    report = json.loads((ROOT / 'reports/harvard_morgan_rf_comparison_v1.json').read_text())
    protocol_path = ROOT / 'configs/harvard_morgan_rf_v1_frozen.json'
    protocol = json.loads(protocol_path.read_text())
    path = ROOT / 'outputs/harvard-morgan-rf-v1/predictions.csv'
    assert report['complete']
    assert sha256(protocol_path) == report['protocol_sha256']
    assert sha256(path) == report['predictions_sha256']
    for name, digest in protocol['input_hashes'].items():
        assert sha256(ROOT / name) == digest, name
    with path.open() as stream:
        rows = list(csv.DictReader(stream))
    ids = np.asarray([r['compound_key'] for r in rows])
    y = np.asarray([int(r['observed_label']) for r in rows])
    with (ROOT / 'outputs/harvard-rn4220-evaluation-v1/predictions.csv').open() as stream:
        reference = {r['compound_key']: int(r['observed_label']) for r in csv.DictReader(stream)}
    assert len(rows) == len(set(ids)) == len(reference) == report['n'] == 225829
    assert y.sum() == report['hits'] == 601
    assert all(reference[key] == label for key, label in zip(ids, y))
    scores = {name: np.asarray([float(r[name]) for r in rows])
              for name in report['rf_members_and_ensemble']}
    np.testing.assert_array_equal(scores['morgan_rf_3'], np.mean(
        [scores[f'morgan_rf_seed{seed}'] for seed in protocol['seeds']], axis=0))
    for name, expected in report['rf_members_and_ensemble'].items():
        values = scores[name]
        assert np.isfinite(values).all() and ((values >= 0) & (values <= 1)).all()
        assert abs(average_precision_score(y, values) - expected['average_precision']) < 1e-12
        assert abs(roc_auc_score(y, values) - expected['roc_auc']) < 1e-12
        order = np.lexsort((ids, -values))
        for k in protocol['top_k']:
            top = expected['top_k'][str(k)]
            assert int(y[order[:k]].sum()) == top['hits']
            cutoff = values[order[k - 1]]
            above, tied = values > cutoff, values == cutoff
            precision = (y[above].sum() + (k - above.sum()) * y[tied].mean()) / k
            assert abs(precision - top['tie_averaged_precision']) < 1e-12
    print(json.dumps({'status': 'passed', 'n': len(rows), 'hits': int(y.sum()),
                      'models': len(scores), 'top_k_checks': 12,
                      'training_started': False, 'files_modified': False}))


if __name__ == '__main__':
    main()
