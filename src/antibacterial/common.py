import hashlib
import importlib.metadata
import json
import platform
import resource
import sys
from pathlib import Path

ROOT = Path(__file__).resolve().parents[2]


def sha256(path):
    return hashlib.sha256(Path(path).read_bytes()).hexdigest()


def save_json(path, value):
    path = Path(path)
    path.parent.mkdir(parents=True, exist_ok=True)
    path.write_text(json.dumps(value, indent=2, ensure_ascii=False) + "\n")


def runtime():
    return {"python": sys.version, "platform": platform.platform(), "architecture": platform.machine(),
            "packages": {name: importlib.metadata.version(name) for name in ("rdkit", "numpy", "scipy", "scikit-learn", "pyarrow")},
            "peak_rss_mib": resource.getrusage(resource.RUSAGE_SELF).ru_maxrss / (1024**2 if sys.platform == "darwin" else 1024)}


def source_hashes():
    return {str(p.relative_to(ROOT)): sha256(p) for p in sorted((ROOT / "src").rglob("*.py"))}
