"""Audit Harvard RN4220 screen without model scoring or changing frozen datasets.

Use the deposited hit rule, retain original readings, and exclude ambiguous parents.
Identity-only and scaffold-excluded inventories are different evaluation populations.
"""
import collections
import csv
import gzip
import json
import math
import resource
import time
from pathlib import Path

from rdkit import RDLogger
from antibacterial.chemistry import standardize
from antibacterial.common import sha256
from link_npass_domain_v1 import protected_sets, KEYS

ROOT = Path(__file__).resolve().parents[1]
RAW = ROOT / 'data/raw/test_dataset_search_v1'
OUT = ROOT / 'data/processed/harvard_rn4220_external_v1'
MEASUREMENTS = ['Absorbance_WT_A', 'Absorbance_WT_B', 'Z-score WT_A',
                'Z-score WT_B', 'Percent Inhibition_WT_A', 'Percent Inhibition_WT_B']


def main():
    t = time.perf_counter()
    OUT.mkdir(exist_ok=True)
    RDLogger.DisableLog('rdApp.*')
    protected, protected_hashes = protected_sets()
    source = json.loads((RAW / 'aid1259311_download_manifest.json').read_text())
    counts = collections.Counter()
    overlap_counts = collections.defaultdict(collections.Counter)
    group_labels = collections.defaultdict(set)
    group_graphs = collections.defaultdict(set)
    group_sids = collections.defaultdict(list)
    cid_labels = collections.defaultdict(set)
    blocked_parents = set()
    with gzip.open(OUT / 'record_audit.jsonl.gz', 'wt') as out:
        for batch in source['batches']:
            path = ROOT / batch['file']
            assert sha256(path) == batch['sha256']
            with path.open() as f:
                for source_line, r in enumerate(csv.DictReader(f), 2):
                    if not r['PUBCHEM_SID']:
                        continue
                    counts['source_records'] += 1
                    cid_labels[r['PUBCHEM_CID']].add(r['PUBCHEM_ACTIVITY_OUTCOME'])
                    counts['author_'+r['PUBCHEM_ACTIVITY_OUTCOME']] += 1
                    flags = []
                    try:
                        values = [float(r[k]) for k in MEASUREMENTS]
                        complete = all(map(math.isfinite, values))
                    except ValueError:
                        complete = False
                    if not complete:
                        flags.append('missing_or_nonfinite_measurements')
                    else:
                        hit = values[0] <= .18 and values[1] <= .18 and values[2] <= -3 and values[3] <= -3
                        if hit != (r['PUBCHEM_ACTIVITY_OUTCOME'] == 'Active'):
                            flags.append('author_rule_mismatch')
                    st = standardize(r['PUBCHEM_EXT_DATASOURCE_SMILES'])
                    flags.extend(st['structure_flags'])
                    v = {**st, 'compound_key': st.get('parent_inchikey')}
                    overlap = {split: [k for k in KEYS if v.get(k) in sets[k]]
                               for split, sets in protected.items()}
                    for split, ks in overlap.items():
                        overlap_counts[split].update(ks)
                    key = st.get('parent_inchikey')
                    if key:
                        group_labels[key].add(r['PUBCHEM_ACTIVITY_OUTCOME'])
                        group_graphs[key].add(st.get('parent_smiles'))
                        group_sids[key].append(r['PUBCHEM_SID'])
                        if flags:
                            blocked_parents.add(key)
                    counts.update(flags)
                    row = {'source_file': batch['file'], 'source_csv_line': source_line,
                           'source': r, 'structure': st, 'measurement_structure_flags': flags,
                           'overlap': overlap}
                    out.write(json.dumps(row, separators=(',', ':'))+'\n')
                    if counts['source_records'] % 20000 == 0:
                        print(f'Audited {counts["source_records"]:,}/329,176 records; '
                              f'{time.perf_counter()-t:.1f}s', flush=True)
    conflicts = {k for k, labels in group_labels.items() if len(labels) != 1}
    ambiguous = {k for k, graphs in group_graphs.items() if len(graphs) != 1}
    cid_conflicts = {k for k, labels in cid_labels.items() if len(labels) != 1}
    seen = set()
    inventories = collections.defaultdict(collections.Counter)
    with gzip.open(OUT / 'record_audit.jsonl.gz', 'rt') as src, \
         gzip.open(OUT / 'identity_unseen.jsonl.gz', 'wt') as ident, \
         gzip.open(OUT / 'scaffold_unseen.jsonl.gz', 'wt') as scaffold:
        for r in map(json.loads, src):
            st = r['structure']
            key = st.get('parent_inchikey')
            if not key or key in seen:
                continue
            if key in blocked_parents or key in conflicts or key in ambiguous:
                continue
            if r['source']['PUBCHEM_CID'] in cid_conflicts:
                continue
            # Any matching identity/connectivity in any of the eight partitions excludes the compound.
            if any(any(k != 'scaffold' for k in ks) for ks in r['overlap'].values()):
                continue
            seen.add(key)
            item = {'compound_key': key, 'parent_smiles': st['parent_smiles'],
                    'connectivity_smiles': st['connectivity_smiles'], 'scaffold': st['scaffold'],
                    'author_activity': int(r['source']['PUBCHEM_ACTIVITY_OUTCOME'] == 'Active'),
                    'source_sids': group_sids[key], 'representative_source_file': r['source_file'],
                    'representative_source_csv_line': r['source_csv_line'],
                    'representative_cid': r['source']['PUBCHEM_CID'],
                    'assay_id': 1259311, 'source_smiles': r['source']['PUBCHEM_EXT_DATASOURCE_SMILES'],
                    'np_status': 'unknown', 'actinomycete_status': 'unknown',
                    'status': 'audited_candidate_not_yet_scored_or_protocol_frozen'}
            ident.write(json.dumps(item, separators=(',', ':'))+'\n')
            inventories['identity_unseen']['compounds'] += 1
            inventories['identity_unseen']['active'] += item['author_activity']
            if not any(r['overlap'].values()):
                scaffold.write(json.dumps(item, separators=(',', ':'))+'\n')
                inventories['scaffold_unseen']['compounds'] += 1
                inventories['scaffold_unseen']['active'] += item['author_activity']
    assert counts['source_records'] == source['rows'] == 329176
    for c in inventories.values():
        c['inactive'] = c['compounds']-c['active']
    report = {'aid': 1259311, 'date': '2026-09-11', 'status': 'candidate_audited_not_scored',
              'source_counts': dict(counts), 'unique_pubchem_cids': len(cid_labels),
              'conflicting_cids': sorted(cid_conflicts), 'parent_label_conflicts': len(conflicts),
              'parent_graph_ambiguities': len(ambiguous),
              'blocked_parents_with_any_qc_flag': len(blocked_parents),
              'inventories': {k: dict(v) for k, v in inventories.items()},
              'overlap_record_counts': {k: dict(v) for k, v in overlap_counts.items()},
              'assay': {'strain': 'S. aureus RN4220 (WT)', 'medium': 'TSB', 'temperature_C': 30,
                        'duration_h': 'approximately 16 (description: 16-18)', 'readout': 'OD600',
                        'replicates': 2, 'compound_transfer_nL': 100, 'final_well_uL': 80,
                        'final_compound_concentration': 'unknown; stock concentration not stated',
                        'hit_rule': 'both OD600 <=0.18 and both plate Z-score <=-3',
                        'limitations': ['OD growth assay does not establish bactericidal action',
                                        'not the Wong 50uM assay; not MIC',
                                        'natural-product/actinomycete origin unknown',
                                        'no plate IDs or stock concentrations in downloaded table',
                                        'does not establish absence from external pretraining corpora']},
              'input_hashes': {**protected_hashes,
                               str((RAW / 'aid1259311_download_manifest.json').relative_to(ROOT)):
                               sha256(RAW / 'aid1259311_download_manifest.json')},
              'output_hashes': {p.name: sha256(p) for p in OUT.glob('*.jsonl.gz')},
              'seconds': time.perf_counter()-t,
              'peak_rss_MiB': resource.getrusage(resource.RUSAGE_SELF).ru_maxrss/1024**2}
    (OUT / 'manifest.json').write_text(json.dumps(report, indent=2)+'\n')
    (ROOT / 'reports/harvard_rn4220_external_audit_v1.json').write_text(json.dumps(report, indent=2)+'\n')
    print(json.dumps({'inventories': report['inventories'], 'seconds': report['seconds'],
                      'peak_rss_MiB': report['peak_rss_MiB']}, indent=2), flush=True)


if __name__ == '__main__':
    main()
