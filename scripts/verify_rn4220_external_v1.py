"""Independent file/identity/label checks of the prepared external inventories."""
import collections
import csv
import gzip
import hashlib
import json
from pathlib import Path
from link_npass_domain_v1 import protected_sets, KEYS

ROOT = Path(__file__).resolve().parents[1]
RAW = ROOT / 'data/raw/test_dataset_search_v1'
OUT = ROOT / 'data/processed/harvard_rn4220_external_v1'


def sha(p):
    return hashlib.sha256(p.read_bytes()).hexdigest()


def main():
    manifest = json.loads((OUT / 'manifest.json').read_text())
    for file, expected in manifest['output_hashes'].items():
        assert sha(OUT / file) == expected
    for file, expected in manifest['input_hashes'].items():
        assert sha(ROOT / file) == expected
    source = {r['SID']: r for r in csv.DictReader((RAW / 'pubchem_aid1259311_concise.csv').open())}
    assert len(source) == manifest['source_counts']['source_records'] == 329176
    protected, _ = protected_sets()
    unions = {k: set().union(*(s[k] for s in protected.values())) for k in KEYS}
    raw_counts = collections.Counter()
    source_labels = {}
    bad_sids = set()
    with gzip.open(OUT / 'record_audit.jsonl.gz', 'rt') as f:
        for r in map(json.loads, f):
            s = r['source']; sid = s['PUBCHEM_SID']
            assert sid not in source_labels
            source_labels[sid] = s['PUBCHEM_ACTIVITY_OUTCOME']
            assert s['PUBCHEM_CID'] == source[sid]['CID']
            assert source_labels[sid] == source[sid]['Activity Outcome']
            raw_counts['records'] += 1
            if r['measurement_structure_flags'] or s['PUBCHEM_CID'] in manifest['conflicting_cids']:
                bad_sids.add(sid)
    assert set(source_labels) == set(source)
    sets = {}
    for name in ['identity_unseen', 'scaffold_unseen']:
        counts = collections.Counter(); ids = set(); emitted_sids = set()
        with gzip.open(OUT / (name+'.jsonl.gz'), 'rt') as f:
            for r in map(json.loads, f):
                assert r['compound_key'] not in ids
                ids.add(r['compound_key'])
                assert bad_sids.isdisjoint(r['source_sids'])
                assert emitted_sids.isdisjoint(r['source_sids'])
                emitted_sids.update(r['source_sids'])
                assert {source_labels[sid] for sid in r['source_sids']} == ({'Active'} if r['author_activity'] else {'Inactive'})
                for k in (KEYS if name == 'scaffold_unseen' else KEYS[:3]):
                    assert r[k] not in unions[k], (name, k, r['compound_key'])
                counts['compounds'] += 1; counts['active'] += r['author_activity']
        counts['inactive'] = counts['compounds'] - counts['active']
        assert dict(counts) == manifest['inventories'][name]
        sets[name] = ids
    assert sets['scaffold_unseen'] <= sets['identity_unseen']
    result = {'status': 'passed', 'source_records': len(source),
              'checks': ['file hashes', 'one-to-one SID and deposited labels', 'unique output parents',
                         'no flagged or conflicting-CID samples retained', 'consistent labels across duplicate samples',
                         'all eight protected partition exclusions', 'strict subset and counts'],
              'limitations': 'This verifies data processing; it does not resolve missing assay concentration or establish experimental validity.'}
    (ROOT / 'reports/harvard_rn4220_external_verification_v1.json').write_text(json.dumps(result, indent=2)+'\n')
    print(json.dumps(result, indent=2))


if __name__ == '__main__':
    main()
