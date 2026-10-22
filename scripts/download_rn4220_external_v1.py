"""Retrieve AID1259311 in bounded batches; never fetch models or make predictions."""
import csv
import hashlib
import json
import subprocess
import time
from pathlib import Path

ROOT = Path(__file__).resolve().parents[1]
RAW = ROOT / 'data/raw/test_dataset_search_v1'
DEST = RAW / 'aid1259311_batches'
URL = 'https://pubchem.ncbi.nlm.nih.gov/rest/pug/assay/aid/1259311/CSV'


def sha(path):
    return hashlib.sha256(path.read_bytes()).hexdigest()


def main():
    start = time.perf_counter()
    DEST.mkdir(exist_ok=True)
    with (RAW / 'pubchem_aid1259311_concise.csv').open() as f:
        rows = list(csv.DictReader(f))
    expected = {r['SID']: r for r in rows}
    assert len(expected) == len(rows) == 329176
    sids = sorted(expected, key=int)
    manifest = {'aid': 1259311, 'url': URL, 'retrieved_date': '2026-09-11',
                'request': 'POST sid comma-separated; batches of 10000',
                'concise_sha256': sha(RAW / 'pubchem_aid1259311_concise.csv'), 'batches': []}
    for offset in range(0, len(sids), 10000):
        subset = sids[offset:offset+10000]
        path = DEST / f'batch_{offset//10000:03d}.csv'
        if not path.exists():
            request = DEST / 'request_sids.txt'
            request.write_text(','.join(subset))
            part = path.with_suffix('.csv.part')
            subprocess.run(['curl', '-L', '--fail', '--retry', '2', '--max-time', '90',
                            '-sS', '--data-urlencode', 'sid@'+str(request), URL,
                            '-o', str(part)], check=True)
            candidate = part
        else:
            candidate = path
        with candidate.open() as f:
            data = [r for r in csv.DictReader(f) if r['PUBCHEM_SID']]
        assert len(data) == len(subset)
        assert {r['PUBCHEM_SID'] for r in data} == set(subset)
        for r in data:
            old = expected[r['PUBCHEM_SID']]
            assert r['PUBCHEM_CID'] == old['CID']
            assert r['PUBCHEM_ACTIVITY_OUTCOME'] == old['Activity Outcome']
        if candidate != path:
            candidate.rename(path)
        manifest['batches'].append({'file': str(path.relative_to(ROOT)), 'rows': len(data),
                                    'bytes': path.stat().st_size, 'sha256': sha(path)})
        print(f'Downloaded and verified {offset+len(subset):,}/{len(sids):,} records; '
              f'{time.perf_counter()-start:.1f}s', flush=True)
    manifest['seconds'] = time.perf_counter()-start
    manifest['rows'] = sum(b['rows'] for b in manifest['batches'])
    (RAW / 'aid1259311_download_manifest.json').write_text(json.dumps(manifest, indent=2)+'\n')


if __name__ == '__main__':
    main()
