"""Verify saved frozen-encoder classifiers against their validation predictions."""
import argparse
import json
from pathlib import Path
import sys

sys.path.insert(0, str(Path(__file__).resolve().parents[1] / "src"))

import joblib
import numpy as np
import pyarrow.parquet as pq
from threadpoolctl import threadpool_limits

from antibacterial.common import ROOT, save_json, sha256
from antibacterial.compare import verify_predictions


def main():
    parser = argparse.ArgumentParser()
    parser.add_argument("--run-id", required=True)
    parser.add_argument("--report", required=True)
    args = parser.parse_args()
    directory = ROOT / "outputs" / args.run_id
    result = json.loads((directory / "results.json").read_text())
    if not result["complete"] or result["pilot_only"] or result["final_test_evaluated"]:
        raise ValueError("Complete development run required")
    processed = ROOT / "data/processed/wong_v0"
    manifest_path = processed / "manifest.json"
    manifest = json.loads(manifest_path.read_text())
    if sha256(manifest_path) != result["dataset_manifest_sha256"]:
        raise ValueError("Data manifest changed")
    for name, expected in manifest["output_hashes"].items():
        if sha256(processed / name) != expected:
            raise ValueError(f"Frozen file changed: {name}")
    cache = ROOT / "outputs" / result["embedding_run"]
    if sha256(cache / "results.json") != result["embedding_results_sha256"]:
        raise ValueError("Embedding provenance changed")
    encoded = json.loads((cache / "results.json").read_text())
    x_path = cache / "validation_embeddings.npy"
    ids_path = cache / "validation_compound_keys.json"
    metadata = encoded["partitions"]["validation"]
    if sha256(x_path) != metadata["cache_sha256"] or sha256(ids_path) != metadata["ids_sha256"]:
        raise ValueError("Validation embedding cache changed")
    validation = pq.read_table(processed / "validation.parquet").to_pylist()
    if json.loads(ids_path.read_text()) != [r["compound_key"] for r in validation]:
        raise ValueError("Embedding order changed")
    x = np.load(x_path, mmap_mode="r", allow_pickle=False)
    checks = {}
    for seed in result["config"]["training_seeds"]:
        model_path = directory / f"frozen_rf_seed{seed}.joblib"
        model = joblib.load(model_path)  # Only models created by this local run.
        if model.get_params() != result["models"][f"frozen_rf_seed{seed}"]["parameters"]:
            raise ValueError("Saved classifier parameters differ")
        prediction_path = directory / f"frozen_rf_seed{seed}_validation.csv"
        saved, _, _ = verify_predictions(prediction_path, validation, result["config"]["top_k"])
        with threadpool_limits(limits=2):
            restored = model.predict_proba(x)[:, 1]
        np.testing.assert_allclose(restored, saved, atol=1e-12, rtol=0)
        checks[str(seed)] = {"model_sha256": sha256(model_path), "predictions_sha256": sha256(prediction_path),
                             "max_restored_score_difference": float(np.max(np.abs(restored - saved)))}
    report = {"run_id": args.run_id, "all_frozen_file_hashes_unchanged": True,
              "restored_models_match_validation_predictions": True, "final_test_evaluated": False,
              "verification_source_sha256": sha256(__file__), "model_checks": checks,
              "encoder_seconds": encoded["total_seconds"], "encoder_peak_rss_mib": encoded["runtime"]["peak_rss_mib"],
              "three_rf_seconds": result["total_seconds"], "rf_peak_rss_mib": result["runtime"]["peak_rss_mib"],
              "embedding_cache_mib": sum((cache / f"{p}_embeddings.npy").stat().st_size for p in ("train", "validation")) / 1024**2}
    save_json(ROOT / args.report, report)
    print(json.dumps(report, indent=2))


if __name__ == "__main__":
    main()
