"""Audit the versioned RN4220/50uM candidate release; do not train or relabel Wong."""
import gzip
import json
import math
import time
from collections import Counter, defaultdict
from pathlib import Path
import numpy as np
import pyarrow.parquet as pq
from rdkit import Chem, RDLogger
from antibacterial.chemistry import standardize
from antibacterial.common import sha256

ROOT = Path(__file__).resolve().parents[1]
RAW = ROOT / 'data/raw/assay_discovery/synthemol_rl'


def main():
    start = time.perf_counter(); RDLogger.DisableLog('rdApp.warning')
    manifest = json.loads((RAW / 'manifest.json').read_text())
    for name, meta in manifest['files'].items():
        assert sha256(RAW / name) == meta['sha256']
    protected = {}
    wong_manifest = json.loads((ROOT / 'data/processed/wong_v0/manifest.json').read_text())
    for split in ('train', 'validation', 'test'):
        p = ROOT / f'data/processed/wong_v0/{split}.parquet'
        assert sha256(p) == wong_manifest['output_hashes'][f'{split}.parquet']
        columns = ['compound_key', 'parent_smiles', 'connectivity_smiles', 'scaffold']
        rows = pq.read_table(p, columns=columns).to_pylist()
        protected[split] = {key: {r[key] for r in rows} for key in columns}
    raw = [json.loads(line) for line in (RAW / 'raw_rows.jsonl').open()]
    means = np.array([r['values']['16h Normalized Mean'] for r in raw], dtype=float)
    assert len(raw) == 10716 and np.isfinite(means).all()
    thresholds = {f'ddof_{d}': float(means.mean()-2*means.std(ddof=d)) for d in (0,1)}
    raw_by_canonical = defaultdict(list)
    for r in raw:
        mol = Chem.MolFromSmiles(r['values']['smiles'])
        if mol is not None:
            raw_by_canonical[Chem.MolToSmiles(mol)].append(r)
    records, identities = [], defaultdict(list)
    label_counts, flags = Counter(), Counter()
    for line in (RAW / 'binarized_rows.jsonl').open():
        row = json.loads(line); v = row['values']; label = v['s_aureus_activity']
        assert label in (0,1)
        label_counts[label] += 1
        st = standardize(v['smiles']); flags.update(st['structure_flags'])
        molecule = Chem.MolFromSmiles(v['smiles'])
        matches = raw_by_canonical.get(Chem.MolToSmiles(molecule), []) if molecule is not None else []
        rec = {'source_row': row['source_row'], 'source_sheet': row['source_sheet'],
               'smiles':v['smiles'], 'author_activity':label, 'structure':st,
               'raw_sheet_rows': [m['source_row'] for m in matches],
               'mean16_values': [m['values']['16h Normalized Mean'] for m in matches]}
        records.append(rec)
        if st['structure_eligible']:
            identities[st['parent_inchikey']].append(rec)
    assert len(records)==10658 and label_counts[1]==1137
    eligible = [r for r in records if r['structure']['structure_eligible']]
    unique = [v[0] for v in identities.values() if len(v)==1]
    keymap={'compound_key':'parent_inchikey','parent_smiles':'parent_smiles','connectivity_smiles':'connectivity_smiles','scaffold':'scaffold'}
    def overlaps(r, split):
        return {key: r['structure'].get(stkey) in protected[split][key] for key,stkey in keymap.items()}
    counts = {split: {key: sum(overlaps(r,split)[key] for r in unique) for key in keymap} for split in protected}
    # Role-neutral inventory. These two pools are alternatives, not assigned train/test sets.
    pools = {}
    for name, splits in [('no_wong_validation_or_test_overlap', ['validation','test']),
                         ('no_overlap_with_any_wong_split', ['train','validation','test'])]:
        pool = [r for r in unique if not any(any(overlaps(r,s).values()) for s in splits)]
        traced = [r for r in pool if r['raw_sheet_rows']]
        pools[name]={'compounds':len(pool),'author_hits':sum(r['author_activity'] for r in pool),
                     'scaffolds':len({r['structure']['scaffold'] for r in pool}),
                     'unmapped_raw_records':len(pool)-len(traced),
                     'raw_traceable_compounds':len(traced),
                     'raw_traceable_author_hits':sum(r['author_activity'] for r in traced)}
    threshold_reconciliation={}
    for name, cutoff in thresholds.items():
        matched = [r for r in records if r['mean16_values']]
        threshold_reconciliation[name]={'threshold':cutoff,'mapped_binary_rows':len(matched),
            'all_raw_rows_agree_with_author_label':sum(all(int(x<cutoff)==r['author_activity'] for x in r['mean16_values']) for r in matched),
            'any_disagreement_rows':sum(any(int(x<cutoff)!=r['author_activity'] for x in r['mean16_values']) for r in matched)}
    dest=ROOT/'data/processed/synthemol_rl_audit_v1';dest.mkdir(exist_ok=True)
    with gzip.open(dest/'records.jsonl.gz','wt') as f:
        for r in records:
            r['wong_structure_overlap']={s:overlaps(r,s) for s in protected}
            f.write(json.dumps(r)+'\n')
    report={'source_manifest_sha256':sha256(RAW/'manifest.json'),'raw_rows':len(raw),
        'processed_author_rows':len(records),'author_hits':label_counts[1],
        'structure_eligible_rows':len(eligible),'unique_eligible_identities':len(identities),
        'duplicate_identity_groups':sum(len(v)>1 for v in identities.values()),
        'duplicate_identity_rows':sum(len(v) for v in identities.values() if len(v)>1),
        'duplicate_identity_conflicting_author_labels':sum(len({r['author_activity'] for r in v})>1 for v in identities.values()),
        'conservative_singleton_identities':len(unique),'structure_flags':dict(flags),
        'singleton_overlap_counts':counts,'role_neutral_candidate_pools':pools,
        'raw16h_mean_summary':{'mean':float(means.mean()),'sd_population':float(means.std()),'min':float(means.min()),'max':float(means.max())},
        'author_label_reconciliation':threshold_reconciliation,
        'no_raw_canonical_match':sum(not r['raw_sheet_rows'] for r in records),
        'normalization':'16h IQM-normalized OD600, not assumed equivalent to Wong control-normalized growth.',
        'measurement_fields':'Raw and normalized R1/R2/mean at both16h and20h, plate and well metadata present.',
        'condition_source':'Article reports RN4220,50uM,LB,16h,37C; workbook columns alone do not encode all these conditions.',
        'roles_frozen':False,'training_started':False,'wong_activity_labels_read':False,
        'limits':['Different author hit rule and IQM normalization; do not pool binary labels or apply Wong 0.2 directly.',
            'Overlap inventories include ACYCLIC scaffold and are conservative.',
            'Metadata dates in plate identifiers may be Excel type conversions, not experimental dates.',
            'No predictions made on new data; no external performance claimed; source origin unknown.'],
        'records_sha256':sha256(dest/'records.jsonl.gz'),'script_sha256':sha256(Path(__file__)),
        'seconds':time.perf_counter()-start}
    (ROOT/'reports/synthemol_rl_audit_v1.json').write_text(json.dumps(report,indent=2)+'\n')
    print(json.dumps(report,indent=2))


if __name__=='__main__':main()
