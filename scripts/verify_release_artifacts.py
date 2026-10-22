"""Read-only SHA-256 verification of the private release payload; stdlib only."""
import hashlib
import json
from pathlib import Path

ROOT = Path(__file__).resolve().parents[1]


def main():
    manifest = json.loads((ROOT / 'docs/artifact_manifest.json').read_text())
    failures = []
    for item in manifest['files']:
        path = ROOT / item['path']
        if not path.resolve().is_relative_to(ROOT.resolve()):
            raise ValueError('Manifest path escapes the repository')
        if not path.is_file():
            failures.append(f"missing: {item['path']}")
            continue
        digest = hashlib.sha256()
        with path.open('rb') as stream:
            for block in iter(lambda: stream.read(1024 * 1024), b''):
                digest.update(block)
        if path.stat().st_size != item['bytes'] or digest.hexdigest() != item['sha256']:
            failures.append(f"changed: {item['path']}")
    if failures:
        raise SystemExit('\n'.join(failures))
    print(json.dumps({'status': 'passed', 'files': len(manifest['files']),
                      'bytes': sum(item['bytes'] for item in manifest['files']),
                      'release': manifest['release'], 'training_started': False}))


if __name__ == '__main__':
    main()
