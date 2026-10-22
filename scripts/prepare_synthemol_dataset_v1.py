"""Prepare assay-separated training data and reserve unexposed evaluation compounds."""
import collections
import gzip
import hashlib
import json
import math
import time
from pathlib import Path
import pyarrow.parquet as pq

ROOT = Path(__file__).resolve().parents[1]
CONFIG = ROOT / 'configs/synthemol_dataset_v1.json'


def digest(p):
    return hashlib.sha256(p.read_bytes()).hexdigest()


def read_gzip(p):
    with gzip.open(p, 'rt') as f:
        return [json.loads(line) for line in f]


def main():
    start = time.perf_counter()
    config = json.loads(CONFIG.read_text())
    audit = json.loads((ROOT / 'reports/synthemol_rl_audit_v1.json').read_text())
    source = ROOT / config['source_audit']
    assert digest(source) == audit['records_sha256']
    rawpath = ROOT / config['source_raw_rows']
    rawmanifest = json.loads((ROOT / 'data/raw/assay_discovery/synthemol_rl/manifest.json').read_text())
    assert digest(rawpath) == rawmanifest['files']['raw_rows.jsonl']['sha256']
    raw = {r['source_row']: r['values'] for r in map(json.loads, rawpath.open())}
    rows = read_gzip(source)
    fields = config['structural_keys']
    input_hashes = {str(p.relative_to(ROOT)): digest(p) for p in (CONFIG, source, rawpath)}
    datasets = {}
    wong = json.loads((ROOT / 'data/processed/wong_v0/manifest.json').read_text())
    for split in ('train', 'validation', 'test'):
        path = ROOT / f'data/processed/wong_v0/{split}.parquet'
        assert digest(path) == wong['output_hashes'][path.name]
        # Identity-only reads, including train; no old target labels are needed.
        datasets['wong_' + split] = pq.read_table(path, columns=fields).to_pylist()
        input_hashes[str(path.relative_to(ROOT))] = digest(path)
    coadd = json.loads((ROOT / 'data/processed/coadd_auxiliary_v1/manifest.json').read_text())
    for split in ('train', 'validation'):
        path = ROOT / f'data/processed/coadd_auxiliary_v1/{split}.jsonl.gz'
        assert digest(path) == coadd['output_sha256'][path.name]
        datasets['coadd_' + split] = [{k:r[k] for k in fields} for r in read_gzip(path)]
        input_hashes[str(path.relative_to(ROOT))] = digest(path)
    sets = {name:{k:{r[k] for r in records} for k in fields} for name,records in datasets.items()}
    def overlaps(record, names):
        return any(record[k] in sets[name][k] for name in names for k in fields)
    duplicates = collections.Counter(r['structure']['parent_inchikey'] for r in rows if r['structure']['structure_eligible'])
    threshold = audit['author_label_reconciliation']['ddof_0']['threshold']
    reasons = collections.Counter(); accepted = []; review = []
    qc_totals = collections.Counter()
    protected = ('wong_validation','wong_test','coadd_validation')
    required = ['16h Raw R1','16h Raw R2','16h Raw Mean','16h Normalized R1','16h Normalized R2','16h Normalized Mean']
    pre_coadd_validation_candidates = 0
    for row in rows:
        st = row['structure']; reason = None
        record = {'compound_key':st.get('parent_inchikey'), **{k:st.get(k) for k in fields if k != 'compound_key'}}
        if not st['structure_eligible']:
            reason = 'structure_review'
        elif duplicates[record['compound_key']] != 1:
            reason = 'duplicate_identity_quarantined'
        elif not row['raw_sheet_rows']:
            reason = 'raw_measurement_mapping_missing'
        elif overlaps(record, ('wong_validation','wong_test')):
            reason = 'wong_evaluation_overlap'
        else:
            measurements = [raw[i] for i in row['raw_sheet_rows']]
            finite = all(isinstance(m.get(k),(float,int)) and math.isfinite(m[k]) for m in measurements for k in required)
            arithmetic = finite and all(abs((m[p+' R1']+m[p+' R2'])/2-m[p+' Mean']) <= config['mean_rounding_absolute_tolerance'] for m in measurements for p in ('16h Raw','16h Normalized'))
            labels = finite and all(int(m['16h Normalized Mean']<threshold) == row['author_activity'] for m in measurements)
            if not finite:
                reason = 'missing_or_nonfinite_measurement'
            elif not arithmetic:
                reason = 'replicate_mean_inconsistent'
            elif not labels:
                reason = 'author_label_raw_mapping_inconsistent'
            else:
                pre_coadd_validation_candidates += 1
                if overlaps(record, ('coadd_validation',)):
                    reason = 'coadd_validation_overlap'
        if reason:
            reasons[reason] += 1
            review.append({'source_row':row['source_row'],'compound_key':record['compound_key'],'reason':reason})
            continue
        disagreement = any((m['16h Normalized R1']<threshold)!=(m['16h Normalized R2']<threshold) for m in measurements)
        borderline = any(abs(m['16h Normalized Mean']-threshold)<=config['borderline_window_for_flag_only'] for m in measurements)
        qc_totals['replicate_disagreement_retained'] += disagreement
        qc_totals['borderline_mean_retained'] += borderline
        record.update(assay_id=config['assay_id'],activity=row['author_activity'],source_sheet=row['source_sheet'],source_row=row['source_row'],raw_sheet_rows=row['raw_sheet_rows'],
            normalized_growth_16h=[{k:m[k] for k in required} for m in measurements],
            replicate_disagreement=disagreement,borderline_mean=borderline,
            np_status='unknown',actinomycete_status='unknown')
        accepted.append(record)
    assert len(accepted)+sum(reasons.values()) == len(rows)
    # Union-find: any shared identity, connectivity or scaffold keeps records together.
    parents = list(range(len(accepted)))
    def find(i):
        while parents[i]!=i:
            parents[i]=parents[parents[i]]; i=parents[i]
        return i
    def union(a,b):
        a,b=find(a),find(b)
        if a!=b: parents[b]=a
    seen = {}
    for i,r in enumerate(accepted):
        for k in fields:
            key=(k,r[k])
            if key in seen: union(i,seen[key])
            else: seen[key]=i
    groups=collections.defaultdict(list)
    for i,r in enumerate(accepted):groups[find(i)].append(r)
    partitions={k:[] for k in ('train','validation','external')}
    exposed_components = 0
    for records in groups.values():
        component_key=min(r['compound_key'] for r in records)
        exposed=any(overlaps(r, datasets.keys()) for r in records)
        h=int(hashlib.sha256(f"{config['split_seed']}:{component_key}".encode()).hexdigest()[:16],16)/2**64
        if exposed:
            exposed_components += 1; split='train'
        else:
            split='external' if h<.2 else 'validation' if h<.3 else 'train'
        for r in records:
            r.update(split=split,component_id=component_key,prior_structure_exposure=exposed)
        partitions[split].extend(records)
    for split,records in partitions.items():
        assert records, f'Empty {split}'
        for k in fields:
            mine={r[k] for r in records}
            for other,others in partitions.items():
                if other!=split: assert not mine & {r[k] for r in others}, (split,other,k)
            for name in (datasets.keys() if split!='train' else protected):
                assert not mine & sets[name][k], (split,name,k)
    dest = ROOT/'data/processed/synthemol_dataset_v1';dest.mkdir(exist_ok=False)
    (dest/'config.json').write_bytes(CONFIG.read_bytes())
    for split,records in [*partitions.items(), ('review',review)]:
        with gzip.open(dest/f'{split}.jsonl.gz','wt') as f:
            for r in records:f.write(json.dumps(r)+'\n')
    # Count assay records separately from unique standardized parent compound identities.
    training = {'wong': datasets['wong_train'],'coadd': datasets['coadd_train'],'synthemol':partitions['train']}
    keys={name:{r['compound_key'] for r in records} for name,records in training.items()}
    unions={}
    for names in [('wong','coadd'),('wong','synthemol'),('wong','coadd','synthemol')]:
        unique=set().union(*(keys[n] for n in names))
        unions['_plus_'.join(names)]={'compound_assay_records':sum(len(training[n]) for n in names),'unique_parent_inchikeys':len(unique)}
    report={'complete':True,'source_author_rows':len(rows),'source_pre_coadd_validation_qc_candidates':pre_coadd_validation_candidates,
        'accepted_compounds':len(accepted),'excluded_by_first_reason':dict(reasons),'measurement_qc_flags_retained':dict(qc_totals),
        'components':len(groups),'components_with_prior_structure_exposure_assigned_train':exposed_components,
        'partitions':{s:{'compounds':len(rs),'author_hits':sum(r['activity'] for r in rs),'components':len({r['component_id'] for r in rs}),
            'replicate_disagreement_retained':sum(r['replicate_disagreement'] for r in rs)} for s,rs in partitions.items()},
        'training_records_by_assay':{k:len(v) for k,v in training.items()},'combined_training_inventory':unions,
        'new_unique_compounds_over_wong_coadd':len(keys['synthemol']-(keys['wong']|keys['coadd'])),
        'overlap_checks_passed':True,'old_wong_activity_labels_read':False,'new_predictions_created':False,'model_training_started':False,
        'input_hashes':input_hashes,'script_sha256':digest(Path(__file__)),
        'output_hashes':{p.name:digest(p) for p in dest.iterdir() if p.is_file()},'seconds':time.perf_counter()-start,
        'limits':['Assay labels remain separate. CO-ADD continuous inhibition is not a binary Wong/SyntheMol activity label.',
            'CO-ADD pretraining did not improve the prior comparison; the three-assay total is available inventory, not an adopted combined model.',
            'External split is unexposed to declared Wong/CO-ADD inputs by these structural keys, not guaranteed unexposed to arbitrary pretraining corpora.',
            'External assay has a distinct normalized endpoint and hit definition; no external performance measured.',
            'Measurement checks validate released summaries; complete plate controls and IQM normalization were not reconstructed.']}
    (dest/'manifest.json').write_text(json.dumps(report,indent=2)+'\n')
    (ROOT/'reports/synthemol_dataset_preparation_v1.json').write_text(json.dumps(report,indent=2)+'\n')
    print(json.dumps({k:report[k] for k in ('accepted_compounds','excluded_by_first_reason','measurement_qc_flags_retained','partitions','combined_training_inventory','new_unique_compounds_over_wong_coadd','seconds')},indent=2))


if __name__=='__main__':main()
