"""Independently recalculate reported metrics from the saved external CSV."""
import csv
import gzip
import json
from pathlib import Path
import numpy as np
from sklearn.metrics import average_precision_score, roc_auc_score
from antibacterial.common import sha256

ROOT = Path(__file__).resolve().parents[1]
OUT = ROOT/'outputs/harvard-rn4220-evaluation-v1'


def main():
    result = json.loads((OUT/'results.json').read_text())
    cfg_path = ROOT/'configs/harvard_rn4220_evaluation_v1_frozen.json'
    cfg = json.loads(cfg_path.read_text())
    assert sha256(cfg_path) == result['protocol_sha256']
    assert sha256(OUT/'predictions.csv') == result['predictions_sha256']
    for p, h in cfg['input_hashes'].items():
        assert sha256(ROOT/p) == h
    with gzip.open(ROOT/'data/processed/harvard_rn4220_external_v1/scaffold_unseen.jsonl.gz', 'rt') as f:
        actual = {r['compound_key']: r['author_activity'] for r in map(json.loads, f)}
    with (OUT/'predictions.csv').open() as f:
        rows = list(csv.DictReader(f))
    ids = [r['compound_key'] for r in rows]; y = np.asarray([int(r['observed_label']) for r in rows])
    assert len(rows) == len(set(ids)) == len(actual) == result['n'] == 225829
    assert y.sum() == result['hits'] == 601
    assert all(actual[k] == label for k,label in zip(ids,y))
    columns = {n: np.asarray([float(r[n]) for r in rows]) for n in list(result['groups'])+list(result['members'])}
    for family in cfg['config']['model_groups'][1:]:
        expected = np.mean([columns[f'{family}_seed{s}'] for s in cfg['config']['seeds']], axis=0)
        np.testing.assert_array_equal(columns[family], expected)
    for name, reported in {**result['groups'], **result['members']}.items():
        scores = columns[name]; assert np.isfinite(scores).all() and ((scores>=0)&(scores<=1)).all()
        assert abs(average_precision_score(y,scores)-reported['average_precision']) < 1e-12
        assert abs(roc_auc_score(y,scores)-reported['roc_auc']) < 1e-12
        order = np.lexsort((np.asarray(ids),-scores))
        for k in cfg['config']['top_k']:
            top = reported['top_k'][str(k)]
            assert int(y[order[:k]].sum()) == top['hits']
            cutoff = scores[order[k-1]]; above = scores>cutoff; tied = scores==cutoff
            expected = (y[above].sum()+(k-above.sum())*y[tied].mean())/k
            assert abs(float(expected)-top['tie_averaged_precision']) < 1e-12
    receipt = {'status': 'passed', 'n': len(rows), 'hits': int(y.sum()),
               'predictions_sha256': sha256(OUT/'predictions.csv'),
               'checks': ['frozen input hashes', 'unique IDs and exact source labels', 'ensemble means',
                          '13 AP and ROC-AUC calculations', '39 Top-K and boundary-tie expectations']}
    (OUT/'verification.json').write_text(json.dumps(receipt,indent=2)+'\n')
    print(json.dumps(receipt), flush=True)


if __name__ == '__main__':
    main()
