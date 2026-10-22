"""Download unchanged author files and record content hashes; never overwrite raw data."""
import hashlib
import json
from datetime import datetime, timezone
from pathlib import Path
from urllib.request import Request, urlopen

ROOT = Path(__file__).resolve().parents[1]
RAW = ROOT / "data" / "raw" / "wong2024"
MANIFEST = RAW / "manifest.json"


def get(url):
    request = Request(url, headers={"User-Agent": "antibacterial-public-data-research/0.1"})
    with urlopen(request, timeout=90) as response:
        return response.read()


def main():
    RAW.mkdir(parents=True, exist_ok=True)
    if MANIFEST.exists():
        manifest = json.loads(MANIFEST.read_text())
        for item in manifest["files"]:
            assert hashlib.sha256((RAW / item["name"]).read_bytes()).hexdigest() == item["sha256"]
        print("Existing raw files verified; no download or overwrite.")
        return
    progress = RAW / "download_in_progress.json"
    pinned_path = ROOT / "reports/source_manifest.json"
    pinned = json.loads(pinned_path.read_text()) if pinned_path.exists() else None
    if progress.exists():
        manifest = json.loads(progress.read_text())
        commit = manifest["repository_commit"]
    else:
        commit = pinned["repository_commit"] if pinned else json.loads(get("https://api.github.com/repos/felixjwong/antibioticsai/commits/main"))["sha"]
        manifest = {"doi": "10.1038/s41586-023-06887-8", "repository_commit": commit,
                    "retrieved_at_utc": datetime.now(timezone.utc).isoformat(), "files": []}
    base = f"https://raw.githubusercontent.com/felixjwong/antibioticsai/{commit}"
    sources = {
        "author_train.csv": base + "/working_example/train.csv",
        "author_README.md": base + "/README.md",
        "author_LICENSE": base + "/LICENSE",
        "supplementary_data_1.xlsx": "https://media.springernature.com/original/springer-static/esm/art%3A10.1038%2Fs41586-023-06887-8/MediaObjects/41586_2023_6887_MOESM3_ESM.xlsx",
    }
    completed = {f["name"]: f for f in manifest["files"]}
    expected = {f["name"]: f for f in pinned["files"]} if pinned else {}
    for name, url in sources.items():
        path = RAW / name
        if name in completed:
            assert hashlib.sha256(path.read_bytes()).hexdigest() == completed[name]["sha256"]
            continue
        if path.exists():
            raise RuntimeError(f"Unmanifested existing file; inspect before replacing: {path}")
        data = get(url)
        if name in expected and hashlib.sha256(data).hexdigest() != expected[name]["sha256"]:
            raise RuntimeError(f"Downloaded content differs from pinned source manifest: {name}")
        path.write_bytes(data)
        manifest["files"].append({"name": name, "url": url, "bytes": len(data), "sha256": hashlib.sha256(data).hexdigest()})
        print(f"Downloaded {name}: {len(data):,} bytes", flush=True)
        (RAW / "download_in_progress.json").write_text(json.dumps(manifest, indent=2))
    MANIFEST.write_text(json.dumps(manifest, indent=2) + "\n")
    if not pinned_path.exists():
        pinned_path.parent.mkdir(parents=True, exist_ok=True)
        pinned_path.write_text(MANIFEST.read_text())
    (RAW / "download_in_progress.json").unlink()


if __name__ == "__main__":
    main()
