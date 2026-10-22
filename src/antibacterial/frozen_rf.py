"""Train the predeclared RF on frozen embeddings, using train/validation only."""
import os
for variable in ("OMP_NUM_THREADS", "OPENBLAS_NUM_THREADS", "MKL_NUM_THREADS", "VECLIB_MAXIMUM_THREADS"):
    os.environ.setdefault(variable, "2")

import argparse
import json
import time

import joblib
import numpy as np
import pyarrow.parquet as pq
from sklearn.ensemble import RandomForestClassifier
from threadpoolctl import threadpool_limits

from .baseline import metrics, write_predictions
from .common import ROOT, runtime, save_json, sha256, source_hashes


def main():
    parser = argparse.ArgumentParser()
    parser.add_argument("--embedding-run", required=True)
    parser.add_argument("--run-id", required=True)
    parser.add_argument("--pilot", action="store_true")
    args = parser.parse_args()
    for value in (args.embedding_run, args.run_id):
        if not value or any(c not in "abcdefghijklmnopqrstuvwxyzABCDEFGHIJKLMNOPQRSTUVWXYZ0123456789-_" for c in value):
            raise ValueError("Invalid run id")
    config = json.loads((ROOT / "configs/v0.json").read_text())
    encoder_config = json.loads((ROOT / "configs/chemeleon_v0.json").read_text())
    if config["final_test_evaluation"] or encoder_config["final_test_evaluation"]:
        raise ValueError("Development-only classifier")
    source = ROOT / "data/processed/wong_v0"
    manifest = json.loads((source / "manifest.json").read_text())
    cache = ROOT / "outputs" / args.embedding_run
    embedding = json.loads((cache / "results.json").read_text())
    if not embedding["complete"] or embedding["pilot_only"] or not embedding["encoder_parameters_unchanged"]:
        raise ValueError("Complete frozen embedding cache required")
    if embedding["dataset_manifest_sha256"] != sha256(source / "manifest.json") or manifest["config"] != config:
        raise ValueError("Wrong dataset/configuration")
    if embedding["config"] != encoder_config:
        raise ValueError("Embedding configuration changed")
    rows, features = {}, {}
    for partition in ("train", "validation"):
        path = source / (partition + ".parquet")
        if sha256(path) != manifest["output_hashes"][path.name]:
            raise ValueError("Frozen data changed")
        rows[partition] = pq.read_table(path).to_pylist()
        ids_path = cache / f"{partition}_compound_keys.json"
        feature_path = cache / f"{partition}_embeddings.npy"
        metadata = embedding["partitions"][partition]
        if sha256(ids_path) != metadata["ids_sha256"] or sha256(feature_path) != metadata["cache_sha256"]:
            raise ValueError("Embedding cache changed")
        if json.loads(ids_path.read_text()) != [r["compound_key"] for r in rows[partition]]:
            raise ValueError("Embeddings do not align with molecule labels")
        features[partition] = np.load(feature_path, mmap_mode="r", allow_pickle=False)
        if features[partition].shape != (len(rows[partition]), embedding["embedding_dim"]) or not np.isfinite(features[partition]).all():
            raise ValueError("Invalid embedding matrix")
    if args.pilot:
        rng = np.random.default_rng(20260909)
        for partition, count in (("train", 5000), ("validation", 500)):
            indices = rng.choice(len(rows[partition]), count, replace=False)
            rows[partition] = [rows[partition][int(i)] for i in indices]
            features[partition] = features[partition][indices]
    destination = ROOT / "outputs" / args.run_id
    destination.mkdir(parents=True, exist_ok=False)
    start = time.perf_counter()
    result = {"run_id": args.run_id, "pilot_only": args.pilot, "evaluation_partition": "validation", "final_test_evaluated": False,
              "family": "chemeleon_frozen_rf", "config": config, "encoder_config": encoder_config,
              "dataset_manifest_sha256": sha256(source / "manifest.json"), "embedding_run": args.embedding_run,
              "embedding_results_sha256": sha256(cache / "results.json"), "source_code_hashes": source_hashes(), "models": {}}
    save_json(destination / "run_started.json", result)
    labels = {p: np.asarray([r["activity"] for r in values]) for p, values in rows.items()}
    for seed in config["training_seeds"][:1] if args.pilot else config["training_seeds"]:
        parameters = dict(config["rf"])
        if args.pilot:
            parameters["n_estimators"] = 50
        model = RandomForestClassifier(**parameters, random_state=seed, n_jobs=config["max_workers"])
        fit_start = time.perf_counter()
        with threadpool_limits(limits=config["max_workers"]):
            model.fit(features["train"], labels["train"])
            fit_seconds = time.perf_counter() - fit_start
            scores = model.predict_proba(features["validation"])[:, 1]
        measured = {"seed": seed, "fit_seconds": fit_seconds, "parameters": model.get_params(), "runtime": runtime()}
        if measured["runtime"]["peak_rss_mib"] > encoder_config["process_memory_budget_mib"]:
            raise RuntimeError("Classifier exceeded local memory budget")
        if not args.pilot:
            measured["metrics"] = metrics(labels["validation"], scores, [r["compound_key"] for r in rows["validation"]], config["top_k"])
            write_predictions(destination / f"frozen_rf_seed{seed}_validation.csv", rows["validation"], labels["validation"], scores)
            joblib.dump(model, destination / f"frozen_rf_seed{seed}.joblib", compress=3)
        result["models"][f"frozen_rf_seed{seed}"] = measured
        save_json(destination / "results.json", {**result, "complete": False})
        print(f"RF seed {seed}: {fit_seconds:.1f}s fit, {measured['runtime']['peak_rss_mib']:.0f} MiB peak", flush=True)
    result.update({"complete": True, "total_seconds": time.perf_counter() - start, "runtime": runtime()})
    save_json(destination / "results.json", result)
    print(json.dumps({"complete": True, "seconds": result["total_seconds"], "models": {k: v.get("metrics") for k, v in result["models"].items()}}), flush=True)


if __name__ == "__main__":
    main()
