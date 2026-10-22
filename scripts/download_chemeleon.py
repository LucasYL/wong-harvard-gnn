"""Download only the fixed base encoder and the public SMILES list, with checksums."""
import hashlib
import json
import subprocess
from pathlib import Path

ROOT = Path(__file__).resolve().parents[1]
FILES = [
    ("chemeleon_mp.pt", "15460715", 34859448, "6a80b54fdb7de37ef0374d302f01e8ce"),
    ("cleaned_pubchem_1MM.smiles", "15733575", 43931153, "1228db3fab5f6dc93ae652eaffa889c1"),
]


def main():
    destination = ROOT / "data/raw/chemeleon"
    destination.mkdir(parents=True, exist_ok=True)
    records = []
    for name, record, size, md5 in FILES:
        url = f"https://zenodo.org/records/{record}/files/{name}"
        path = destination / name
        if not path.exists():
            partial = destination / (name + ".part")
            subprocess.run(["curl", "--fail", "--location", "--silent", "--show-error", "--retry", "2", "--max-time", "180",
                            url, "-o", str(partial)], check=True)
            content = partial.read_bytes()
            if len(content) != size or hashlib.md5(content).hexdigest() != md5:
                raise ValueError(f"Downloaded content does not match published file: {name}")
            partial.rename(path)
        content = path.read_bytes()
        if len(content) != size or hashlib.md5(content).hexdigest() != md5:
            raise ValueError(f"Existing file differs from pinned artifact: {name}")
        records.append({"file": name, "url": url, "bytes": size, "md5": md5,
                        "sha256": hashlib.sha256(content).hexdigest()})
        print(f"Verified {name}: {size:,} bytes", flush=True)
    manifest = {"purpose": "Base encoder and structural-exposure audit; no target activity labels", "files": records}
    path = destination / "manifest.json"
    if path.exists() and json.loads(path.read_text()) != manifest:
        raise ValueError("Existing manifest differs; version the change")
    path.write_text(json.dumps(manifest, indent=2) + "\n")


if __name__ == "__main__":
    main()
