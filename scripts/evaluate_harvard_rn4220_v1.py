"""Bounded, resumable external evaluation. No training or test-driven selection."""
import argparse
import csv
import gc
import gzip
import hashlib
import json
import os
import time
from datetime import datetime, timezone
from pathlib import Path

import evaluate_external_synthemol_v1 as old
from rdkit import Chem, DataStructs, RDLogger
from rdkit.Chem import rdFingerprintGenerator
from sklearn.metrics import average_precision_score

np, torch = old.np, old.torch
ROOT = Path(__file__).resolve().parents[1]
DATA = ROOT / 'data/processed/harvard_rn4220_external_v1/scaffold_unseen.jsonl.gz'
FROZEN = ROOT / 'configs/harvard_rn4220_evaluation_v1_frozen.json'
OUT = ROOT / 'outputs/harvard-rn4220-evaluation-v1'
PILOT = ROOT / 'reports/harvard_rn4220_inference_pilot_v1.json'
CFG = {'n': 225829, 'hits': 601, 'chunk_size': 1024, 'batch_size': 32,
       'cpu_threads': 2, 'memory_budget_mib': 3072, 'top_k': [20, 50, 100],
       'primary_metric': 'average_precision', 'secondary': 'Top-K hits, precision, enrichment; trapezoidal PR-AUC descriptive',
       'model_groups': ['nearest_neighbour', 'scratch_descriptors_3', 'coadd_transfer_3', 'synthemol_transfer_3'],
       'seeds': [17, 29, 43], 'ensemble': 'equal mean of the three fixed members',
       'tie_policy': 'descending score then ascending compound_key; also report cutoff-tie-averaged precision',
       'bootstrap': {'iterations': 2000, 'seed': 20260911, 'unit': 'Murcko scaffold',
                     'comparison': 'paired AP differences relative to scratch_descriptors_3'},
       'target': 'AID1259311 deposited binary growth-screen labels, NOT Wong labels or MIC',
       'claim_boundary': 'Cross-assay retrospective ranking; concentration unknown; not actinomycete-specific or discovery validation',
       'no_training_tuning_calibration': True, 'metrics_only_after_all_predictions_sealed': True,
       'subsets': 'Only the scaffold-unseen inventory; wider identity inventory is not evaluated',
       'failure_policy': 'Stop on invalid features or inference; do not silently drop compounds',
       'post_evaluation': 'Becomes historical evaluation; cannot be reused as untouched test for adaptive changes'}


def now():
    return datetime.now(timezone.utc).isoformat()


def status(stage, **kw):
    old.save(OUT/'status.json', {'state': 'running', 'stage': stage, 'updated_at_utc': now(),
                               'pid': os.getpid(), **kw})
    print(json.dumps({'stage': stage, **kw}), flush=True)


class Engine:
    def __init__(self, spec):
        self.spec = spec
        self.generator = old.rdNormalizedDescriptors.RDKit2DNormalized()
        self.base = json.loads((ROOT/'configs/v0.json').read_text())
        self.fp_generator = rdFingerprintGenerator.GetMorganGenerator(**self.base['fingerprint'])
        rows = old.pq.read_table(ROOT/'data/processed/wong_v0/train.parquet',
                                columns=['parent_smiles', 'activity']).to_pylist()
        self.train_fps = [self.fp_generator.GetFingerprint(Chem.MolFromSmiles(r['parent_smiles'])) for r in rows]
        self.y = np.asarray([r['activity'] for r in rows])
        self.networks = []
        self.names = ['nearest_neighbour']
        for entry in spec['models']:
            network = old.model(json.loads((ROOT/'configs/chemprop_descriptors_v1.json').read_text()))
            network.load_state_dict(torch.load(ROOT/entry['checkpoint'], map_location='cpu', weights_only=False)['state_dict'], strict=True)
            network.eval()
            self.networks.append(network)
            self.names.append(f"{entry['family']}_seed{entry['seed']}")

    def predict(self, rows, report=False):
        t = time.perf_counter(); features = []; imputed = []
        for i, r in enumerate(rows):
            x, missing = old.vector(self.generator, r['parent_smiles'])
            features.append(x)
            if missing:
                imputed.append({'compound_key': r['compound_key'], 'indices': missing})
            if report and (i+1) % 256 == 0:
                print(f'  Descriptors {i+1}/{len(rows)}; {time.perf_counter()-t:.1f}s', flush=True)
        features = np.stack(features)
        descriptor_seconds = time.perf_counter()-t; t = time.perf_counter()
        fps = [self.fp_generator.GetFingerprint(Chem.MolFromSmiles(r['parent_smiles'])) for r in rows]
        scores, _, similarity, _ = old.neighbour_predict(self.train_fps, fps, self.y, self.base['nearest_neighbours'])
        nn_seconds = time.perf_counter()-t; t = time.perf_counter()
        matrix = np.empty((len(rows), len(self.names)), dtype=np.float64)
        matrix[:, 0] = scores
        # Placeholder y is required by the Dataset interface but never supplied to forward().
        loader = old.make_loader(old.dataset([{**r, 'activity': 0} for r in rows], features), CFG['batch_size'])
        offset = 0
        with torch.inference_mode():
            for graph, vd, xd, *_ in loader:
                n = len(xd)
                for j, network in enumerate(self.networks, 1):
                    matrix[offset:offset+n, j] = network(graph, vd, xd).numpy().reshape(-1)
                offset += n
        assert offset == len(rows) and np.isfinite(matrix).all()
        assert ((matrix >= 0) & (matrix <= 1)).all()
        old.budget(CFG)
        return matrix, features, np.asarray(similarity), imputed, {
            'descriptors': descriptor_seconds, 'nearest_neighbour': nn_seconds,
            'nine_neural_members': time.perf_counter()-t}


def ap_cache(scores):
    order = np.argsort(-np.asarray(scores), kind='stable')
    sorted_scores = np.asarray(scores)[order]
    starts = np.r_[0, np.flatnonzero(np.diff(sorted_scores)) + 1]
    return order, starts


def weighted_ap(y, weights, cache):
    order, starts = cache
    positives = np.add.reduceat((weights*y)[order], starts)
    totals = np.add.reduceat(weights[order], starts)
    cum = np.cumsum(totals)
    precision = np.divide(np.cumsum(positives), cum, out=np.zeros_like(cum, dtype=float), where=cum>0)
    return float(np.dot(positives, precision)/positives.sum())


def pilot():
    assert not PILOT.exists(), 'Existing pilot must be inspected, not overwritten'
    spec = old.checked(); engine = Engine(spec)
    rows = old.read_rows('validation', labels=False)
    rows = sorted(rows, key=lambda r: hashlib.sha256(r['compound_key'].encode()).hexdigest())[:512]
    started = time.perf_counter()
    matrix, _, _, _, timings = engine.predict(rows, report=True)
    errors = {}
    paths = [ROOT/'outputs/development-20260909-numeric-fix/nearest_neighbour_validation.csv'] + [ROOT/e['validation_predictions'] for e in spec['models']]
    for j, path in enumerate(paths):
        with path.open() as f:
            saved = {r['compound_key']: float(r['predicted_score']) for r in csv.DictReader(f)}
        error = float(np.max(np.abs(matrix[:, j]-[saved[r['compound_key']] for r in rows])))
        assert error <= (1e-12 if j == 0 else 1e-6), (engine.names[j], error)
        errors[engine.names[j]] = error
    # Meaningful check: ties and absent bootstrap observations must agree with sklearn.
    for seed in range(5):
        rng = np.random.default_rng(seed); y = rng.integers(2, size=200)
        scores = rng.integers(10, size=200)/10; weights = rng.integers(4, size=200).astype(float)
        assert abs(weighted_ap(y, weights, ap_cache(scores))-average_precision_score(y, scores, sample_weight=weights)) < 1e-12
    seconds = time.perf_counter()-started
    old.save(PILOT, {'created_at_utc': now(), 'n': 512, 'source': 'old Wong validation, selected by key hash; no Harvard predictions',
                     'seconds': seconds, 'stage_seconds': timings, 'reproduction_errors': errors,
                     'weighted_ap_check': 'passed including score ties and zero sample weights',
                     'estimated_prediction_minutes': sum(timings.values())/512*CFG['n']/60,
                     'runtime': old.machine_runtime(), 'runner_sha256': old.sha256(Path(__file__)), 'config': CFG})
    print(PILOT.read_text(), flush=True)


def freeze():
    assert not FROZEN.exists()
    pilot_result = json.loads(PILOT.read_text())
    assert pilot_result['runner_sha256'] == old.sha256(Path(__file__)) and pilot_result['config'] == CFG
    audit = json.loads((DATA.parent/'manifest.json').read_text())
    assert old.sha256(DATA) == audit['output_hashes'][DATA.name]
    assert audit['inventories']['scaffold_unseen'] == {'compounds': CFG['n'], 'active': CFG['hits'], 'inactive': CFG['n']-CFG['hits']}
    assert json.loads((ROOT/'reports/harvard_rn4220_external_verification_v1.json').read_text())['status'] == 'passed'
    inputs = [DATA, DATA.parent/'manifest.json', PILOT, Path(__file__), ROOT/'scripts/verify_rn4220_external_v1.py',
              ROOT/'reports/harvard_rn4220_external_verification_v1.json']
    old.save(FROZEN, {'created_at_utc': now(), 'config': CFG, 'models': old.checked(),
                     'input_hashes': {str(p.relative_to(ROOT)): old.sha256(p) for p in inputs}})
    print('Frozen Harvard evaluation: cohort, models, metrics, ties, bootstrap, inputs and code.', flush=True)


def checked():
    cfg = json.loads(FROZEN.read_text())
    assert cfg['config'] == CFG and cfg['models'] == old.checked()
    for p, h in cfg['input_hashes'].items():
        assert old.sha256(ROOT/p) == h, p
    return cfg


def batches():
    batch = []
    with gzip.open(DATA, 'rt') as f:
        for r in map(json.loads, f):
            batch.append({k: r[k] for k in ['compound_key', 'parent_smiles', 'scaffold']})
            if len(batch) == CFG['chunk_size']:
                yield batch; batch = []
        if batch:
            yield batch


def evaluate():
    cfg = checked()
    assert not (OUT/'results.json').exists(), 'Completed evaluation cannot be rerun'
    OUT.mkdir(exist_ok=True); (OUT/'chunks').mkdir(exist_ok=True)
    # Prevent concurrent writers. The OS releases this lock when the process exits.
    import fcntl
    lock = (OUT/'run.lock').open('w'); fcntl.flock(lock, fcntl.LOCK_EX | fcntl.LOCK_NB)
    started = time.perf_counter(); completed = 0; engine = None
    try:
        status('checking_inputs_and_loading_models', total=CFG['n'])
        engine = Engine(cfg['models'])
        for b, rows in enumerate(batches()):
            path = OUT/'chunks'/f'{b:04d}.npz'; meta = path.with_suffix('.json')
            ids = [r['compound_key'] for r in rows]
            if meta.exists():
                m = json.loads(meta.read_text())
                assert m['protocol_sha256'] == old.sha256(FROZEN) and m['sha256'] == old.sha256(path)
                with np.load(path, allow_pickle=False) as saved:
                    assert saved['ids'].tolist() == ids
                completed += len(rows)
                continue
            status('predicting_chunk', completed=completed, total=CFG['n'], chunk=b+1, chunks=221,
                   peak_rss_mib=old.machine_runtime()['peak_rss_mib'])
            scores, features, sim, imputed, timings = engine.predict(rows, report=True)
            tmp = path.with_suffix('.npz.part')
            with tmp.open('wb') as f:
                np.savez_compressed(f, ids=np.asarray(ids), scores=scores, descriptors=features, similarity=sim)
            os.replace(tmp, path)
            old.save(meta, {'protocol_sha256': old.sha256(FROZEN), 'sha256': old.sha256(path),
                            'rows': len(rows), 'model_order': engine.names, 'imputed': imputed,
                            'seconds': timings, 'finished_at_utc': now()})
            completed += len(rows)
            status('chunk_saved', completed=completed, total=CFG['n'], chunk=b+1, chunks=221,
                   elapsed_this_process_seconds=time.perf_counter()-started,
                   peak_rss_mib=old.machine_runtime()['peak_rss_mib'])
            gc.collect()
        assert completed == CFG['n']
        status('sealing_predictions', completed=completed, total=CFG['n'])
        chunk_meta = sorted((OUT/'chunks').glob('*.json'))
        old.save(OUT/'predictions_sealed.json', {'created_at_utc': now(), 'protocol_sha256': old.sha256(FROZEN),
                    'n': completed, 'model_order': engine.names, 'labels_used_in_forward': False,
                    'chunk_receipts': {str(p.relative_to(OUT)): old.sha256(p) for p in chunk_meta}})
        del engine; gc.collect()
        finish()
    except Exception as exc:
        old.save(OUT/'status.json', {'state': 'failed', 'error': str(exc), 'completed': completed,
                                   'updated_at_utc': now(), 'pid': os.getpid()})
        raise


def finish():
    checked(); seal = json.loads((OUT/'predictions_sealed.json').read_text())
    ids, arrays, similarities = [], [], []
    for rel, digest in seal['chunk_receipts'].items():
        meta = OUT/rel; assert old.sha256(meta) == digest
        m = json.loads(meta.read_text()); path = meta.with_suffix('.npz')
        assert old.sha256(path) == m['sha256']
        with np.load(path, allow_pickle=False) as saved:
            ids.extend(saved['ids'].tolist()); arrays.append(saved['scores']); similarities.append(saved['similarity'])
    matrix = np.concatenate(arrays); similarity = np.concatenate(similarities)
    del arrays, similarities
    with gzip.open(DATA, 'rt') as f:
        source = list(map(json.loads, f))
    assert ids == [r['compound_key'] for r in source] and len(set(ids)) == CFG['n']
    y = np.asarray([r['author_activity'] for r in source]); assert y.sum() == CFG['hits']
    predictions = {n: matrix[:, j] for j, n in enumerate(seal['model_order'])}
    for family in CFG['model_groups'][1:]:
        predictions[family] = old.mean_scores([predictions[f'{family}_seed{s}'] for s in CFG['seeds']])
    status('calculating_metrics', completed=CFG['n'], total=CFG['n'])
    metrics = {n: old.all_metrics(y, s, ids, CFG['top_k']) for n, s in predictions.items()}
    # Persist all results by identity before resampling. No feature changes or model selection here.
    with (OUT/'predictions.csv').open('w', newline='') as f:
        writer = csv.writer(f); writer.writerow(['compound_key', 'observed_label', 'nearest_wong_train_similarity', *predictions])
        writer.writerows([key, int(y[i]), float(similarity[i]), *[float(s[i]) for s in predictions.values()]] for i,key in enumerate(ids))
    _, groups = np.unique([r['scaffold'] for r in source], return_inverse=True)
    n_groups = int(groups.max()+1); rng = np.random.default_rng(CFG['bootstrap']['seed'])
    caches = {n: ap_cache(predictions[n]) for n in CFG['model_groups']}
    delta = {n: [] for n in CFG['model_groups'] if n != 'scratch_descriptors_3'}
    for iteration in range(CFG['bootstrap']['iterations']):
        weights = np.bincount(rng.integers(n_groups, size=n_groups), minlength=n_groups)[groups].astype(float)
        if (weights*y).sum() == 0 or (weights*(1-y)).sum() == 0:
            continue
        ap = {n: weighted_ap(y, weights, c) for n,c in caches.items()}
        for n in delta:
            delta[n].append(ap[n]-ap['scratch_descriptors_3'])
        if (iteration+1) % 100 == 0:
            status('scaffold_bootstrap', iteration=iteration+1, iterations=CFG['bootstrap']['iterations'], completed=CFG['n'], total=CFG['n'])
    result = {'complete': True, 'finished_at_utc': now(), 'protocol_sha256': old.sha256(FROZEN),
              'n': len(ids), 'hits': int(y.sum()), 'scaffold_groups': n_groups,
              'groups': {n: metrics[n] for n in CFG['model_groups']},
              'members': {n: m for n,m in metrics.items() if '_seed' in n},
              'paired_ap_difference_vs_scratch': {n: {'observed': metrics[n]['average_precision']-metrics['scratch_descriptors_3']['average_precision'],
                   'percentile_95_interval': np.quantile(d, [.025, .975]).tolist(), 'valid_replicates': len(d)} for n,d in delta.items()},
              'runtime': old.machine_runtime(), 'predictions_sha256': old.sha256(OUT/'predictions.csv'),
              'limits': ['Fixed-model, scaffold-cluster conditional bootstrap, not training uncertainty.',
                         'Different assay; unknown concentrations; no actinomycete-specific or discovery claim.',
                         'Historical evaluation after this run; no adaptive tuning on these results.']}
    checked(); old.save(OUT/'results.json', result)
    old.save(ROOT/'reports/harvard_rn4220_model_comparison_v1.json', result)
    old.save(OUT/'status.json', {'state': 'complete', 'updated_at_utc': now(), 'completed': len(ids), 'total': len(ids)})
    print(json.dumps(result['groups'], indent=2), flush=True)


if __name__ == '__main__':
    parser = argparse.ArgumentParser(); parser.add_argument('action', choices=['pilot', 'freeze', 'run'])
    action = parser.parse_args().action
    torch.set_num_threads(2); torch.set_num_interop_threads(1); torch.use_deterministic_algorithms(True)
    RDLogger.DisableLog('rdApp.warning')
    {'pilot': pilot, 'freeze': freeze, 'run': evaluate}[action]()
