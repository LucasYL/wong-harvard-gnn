"""Versioned structure-only descriptor cache for train and validation partitions."""
import argparse
import importlib.metadata
import json
import time
from pathlib import Path

import numpy as np
import pyarrow.parquet as pq
from descriptastorus.descriptors import rdNormalizedDescriptors

from .common import ROOT, runtime, save_json, sha256

CONFIG = ROOT / "configs/chemprop_descriptors_v1.json"
CACHE = ROOT / "data/processed/wong_descriptors_v1"
PROCESSED = ROOT / "data/processed/wong_v0"


def identity():
    return {"config_sha256": sha256(CONFIG), "descriptor_code_sha256": sha256(Path(__file__)),
            "dataset_manifest_sha256": sha256(PROCESSED / "manifest.json"),
            "packages": {p: importlib.metadata.version(p) for p in ("descriptastorus", "rdkit", "numpy")}}


def read_rows(partition, labels=False):
    if partition not in ("train", "validation"):
        raise ValueError("Descriptor development permits only train and validation")
    manifest = json.loads((PROCESSED / "manifest.json").read_text())
    path = PROCESSED / f"{partition}.parquet"
    if sha256(path) != manifest["output_hashes"][path.name]:
        raise ValueError("Dataset content changed")
    columns = None if labels else ["compound_key", "parent_smiles"]
    return pq.read_table(path, columns=columns).to_pylist()


def vector(generator, smiles):
    result = generator.process(smiles)
    if result is None or len(result) != 201 or not result[0]:
        raise ValueError(f"Descriptor calculation failed: {smiles}")
    values = np.asarray(result[1:], dtype=np.float64)
    missing = np.flatnonzero(~np.isfinite(values))
    values[missing] = 0.0
    if np.any((values < 0) | (values > 1)):
        raise ValueError("Normalized descriptor outside [0, 1]")
    return values.astype(np.float32), missing.tolist()


def load_features(partition, rows):
    manifest = json.loads((CACHE / "manifest.json").read_text())
    if not manifest["complete"] or manifest["identity"] != identity():
        raise ValueError("Descriptor cache identity changed or incomplete")
    item = manifest["partitions"][partition]
    for name, digest in item["hashes"].items():
        if sha256(CACHE / name) != digest:
            raise ValueError("Descriptor cache file changed")
    keys = json.loads((CACHE / f"{partition}_keys.json").read_text())
    if keys != [r["compound_key"] for r in rows]:
        raise ValueError("Descriptor row identity/order mismatch")
    values = np.load(CACHE / f"{partition}.npy", allow_pickle=False)
    if values.shape != (len(rows), 200) or not np.isfinite(values).all():
        raise ValueError("Invalid descriptor matrix")
    return values


def main():
    parser = argparse.ArgumentParser()
    parser.add_argument("--pilot", action="store_true")
    args = parser.parse_args()
    start, original = time.perf_counter(), identity()
    generator = rdNormalizedDescriptors.RDKit2DNormalized()
    columns = [c[0] for c in generator.GetColumns()[1:]]
    if len(columns) != 200 or len(set(columns)) != 200:
        raise ValueError("Unexpected descriptor definition")
    all_rows = {p: read_rows(p) for p in ("train", "validation")}
    if args.pilot:
        rows = all_rows["train"]
        rng = np.random.default_rng(20260909)
        selected = [rows[i] for i in rng.choice(len(rows), 256, replace=False)]
        times, missing = [], 0
        for row in selected:
            t = time.perf_counter()
            _, absent = vector(generator, row["parent_smiles"])
            times.append(time.perf_counter() - t)
            missing += len(absent)
        result = {"identity": original, "n": len(selected), "seconds": sum(times),
                  "mean_seconds_per_molecule": float(np.mean(times)), "max_seconds": max(times),
                  "missing_values": missing, "runtime": runtime(),
                  "estimated_full_minutes": float(np.mean(times) * sum(map(len, all_rows.values())) / 60),
                  "no_performance_metrics": True, "final_test_read": False}
        save_json(ROOT / "reports/descriptor_pilot_v1.json", result)
        print(json.dumps(result), flush=True)
        return
    if (CACHE / "manifest.json").exists():
        for p, rows in all_rows.items():
            load_features(p, rows)
        print("Verified existing complete descriptor cache; no recomputation", flush=True)
        return
    CACHE.mkdir(parents=True, exist_ok=True)
    result = {"identity": original, "columns": columns, "partitions": {}, "complete": False,
              "final_test_read": False, "scaling": "Fixed descriptastorus normalization; no target-dataset fit"}
    for partition, rows in all_rows.items():
        values = np.empty((len(rows), 200), dtype=np.float32)
        imputed = []
        for i, row in enumerate(rows):
            values[i], absent = vector(generator, row["parent_smiles"])
            if absent:
                imputed.append({"compound_key": row["compound_key"], "columns": [columns[j] for j in absent]})
            if (i + 1) % 1000 == 0:
                progress = {"partition": partition, "completed": i + 1, "total": len(rows),
                            "seconds": time.perf_counter() - start, "peak_rss_mib": runtime()["peak_rss_mib"]}
                save_json(CACHE / "progress.json", progress)
                print(json.dumps(progress), flush=True)
        np.save(CACHE / f"{partition}.npy", values, allow_pickle=False)
        save_json(CACHE / f"{partition}_keys.json", [r["compound_key"] for r in rows])
        save_json(CACHE / f"{partition}_imputation.json", imputed)
        names = [f"{partition}.npy", f"{partition}_keys.json", f"{partition}_imputation.json"]
        result["partitions"][partition] = {"n": len(rows), "imputed_compounds": len(imputed),
                                           "hashes": {n: sha256(CACHE / n) for n in names}}
    if original != identity():
        raise ValueError("Input or code changed during descriptor generation")
    result.update(complete=True, seconds=time.perf_counter() - start, runtime=runtime())
    save_json(CACHE / "manifest.json", result)
    print(json.dumps({k: result[k] for k in ("complete", "seconds", "runtime", "partitions")}), flush=True)


if __name__ == "__main__":
    main()
