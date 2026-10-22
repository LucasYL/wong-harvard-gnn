"""Development evaluation only: this module never reads the final test partition."""
import os

for variable in ("OMP_NUM_THREADS", "OPENBLAS_NUM_THREADS", "MKL_NUM_THREADS", "VECLIB_MAXIMUM_THREADS"):
    os.environ.setdefault(variable, "2")

import argparse
import csv
import json
import time
from datetime import datetime, timezone

import joblib
import numpy as np
import pyarrow.parquet as pq
from rdkit import Chem, DataStructs
from rdkit.Chem import rdFingerprintGenerator
from sklearn.ensemble import RandomForestClassifier
from sklearn.metrics import average_precision_score, roc_auc_score, brier_score_loss
from threadpoolctl import threadpool_limits

from .common import ROOT, runtime, save_json, sha256, source_hashes


def checked_probability(value):
    """Correct only floating-point overshoot; reject real numerical/model failures."""
    if not np.isfinite(value) or value < -1e-12 or value > 1 + 1e-12:
        raise ValueError(f"Probability is outside the valid range: {value}")
    return min(1.0, max(0.0, float(value)))


def metrics(labels, scores, ids, top_k):
    labels, scores = np.asarray(labels), np.asarray(scores)
    if not np.isfinite(scores).all() or len(set(labels)) != 2:
        raise ValueError("Metrics require finite scores and both observed classes.")
    order = np.lexsort((np.asarray(ids), -scores))
    prevalence = float(labels.mean())
    result = {"n": len(labels), "hits": int(labels.sum()), "hit_rate": prevalence,
              "average_precision": float(average_precision_score(labels, scores)),
              "roc_auc": float(roc_auc_score(labels, scores)),
              "brier_score": float(brier_score_loss(labels, scores)), "top_k": {}}
    for k in top_k:
        if k > len(labels):
            continue
        cutoff = scores[order[k - 1]]
        above, tied = scores > cutoff, scores == cutoff
        precision = float(labels[order[:k]].mean())
        # Also report expectation over the cutoff tie group; no label-based tie breaking.
        expected_hits = float(labels[above].sum() + (k - above.sum()) * labels[tied].mean())
        result["top_k"][str(k)] = {"hits": int(labels[order[:k]].sum()), "precision": precision,
                                    "enrichment": precision / prevalence, "cutoff_tie_size": int(tied.sum()),
                                    "tie_averaged_precision": expected_hits / k,
                                    "tie_averaged_enrichment": expected_hits / k / prevalence}
    return result


def fingerprints(rows, specification):
    generator = rdFingerprintGenerator.GetMorganGenerator(**specification)
    fps = [generator.GetFingerprint(Chem.MolFromSmiles(r["parent_smiles"])) for r in rows]
    array = np.zeros((len(rows), specification["fpSize"]), dtype=np.uint8)
    for index, fp in enumerate(fps):
        DataStructs.ConvertToNumpyArray(fp, array[index])
    return fps, array


def neighbour_predict(train_fps, query_fps, train_y, neighbours):
    predictions, closest, similarities, used = [], [], [], []
    y = np.asarray(train_y)
    for count, fp in enumerate(query_fps, 1):
        values = np.asarray(DataStructs.BulkTanimotoSimilarity(fp, train_fps))
        cutoff = np.partition(values, -min(neighbours, len(values)))[-min(neighbours, len(values))]
        selected = values >= cutoff  # include every tied neighbour at the boundary
        weight = values[selected].sum()
        prediction = float(np.dot(values[selected], y[selected]) / weight) if weight > 0 else float(y.mean())
        predictions.append(checked_probability(prediction))
        closest.append(int(np.argmax(values)))
        similarities.append(float(values.max()))
        used.append(int(selected.sum()))
        if count % 1000 == 0:
            print(f"Nearest-neighbour predictions: {count:,}/{len(query_fps):,}", flush=True)
    return np.asarray(predictions), np.asarray(closest), np.asarray(similarities), used


def write_predictions(path, rows, labels, scores, extra=None):
    records = []
    for i, row in enumerate(rows):
        records.append({"compound_key": row["compound_key"], "parent_smiles": row["parent_smiles"],
                        "source_rows": json.dumps(row["source_rows"]), "observed_label": int(labels[i]),
                        "predicted_score": float(scores[i]), **({k: v[i] for k, v in extra.items()} if extra else {})})
    with path.open("w", newline="") as handle:
        writer = csv.DictWriter(handle, fieldnames=list(records[0]))
        writer.writeheader()
        writer.writerows(records)


def main():
    parser = argparse.ArgumentParser()
    parser.add_argument("--pilot", action="store_true")
    parser.add_argument("--run-id", required=True)
    args = parser.parse_args()
    if not args.run_id or any(c not in "abcdefghijklmnopqrstuvwxyzABCDEFGHIJKLMNOPQRSTUVWXYZ0123456789-_" for c in args.run_id):
        raise ValueError("run-id must contain only letters, digits, hyphens and underscores")
    config_path = ROOT / "configs/v0.json"
    config = json.loads(config_path.read_text())
    if config["final_test_evaluation"] is not False:
        raise ValueError("This command is for development only and never evaluates final test data.")
    destination = ROOT / "outputs" / args.run_id
    destination.mkdir(parents=True, exist_ok=False)  # runs are append-only, never overwritten
    processed = ROOT / "data/processed/wong_v0"
    dataset_manifest = json.loads((processed / "manifest.json").read_text())
    if dataset_manifest["config"] != config:
        raise ValueError("Config changed since dataset preparation; document and version before running.")
    # Only development file hashes are checked here. Final test rows/labels are never opened.
    for name in ("train.parquet", "validation.parquet"):
        if sha256(processed / name) != dataset_manifest["output_hashes"][name]:
            raise ValueError(f"Dataset content hash changed: {name}")
    train = pq.read_table(processed / "train.parquet").to_pylist()
    validation = pq.read_table(processed / "validation.parquet").to_pylist()
    if args.pilot:
        rng = np.random.default_rng(20260908)
        train = [train[int(i)] for i in rng.choice(len(train), min(5000, len(train)), replace=False)]
        validation = [validation[int(i)] for i in rng.choice(len(validation), min(500, len(validation)), replace=False)]
    start = time.perf_counter()
    train_fps, train_x = fingerprints(train, config["fingerprint"])
    validation_fps, validation_x = fingerprints(validation, config["fingerprint"])
    train_y = np.array([r["activity"] for r in train])
    validation_y = np.array([r["activity"] for r in validation])
    keys = [r["compound_key"] for r in validation]
    result = {"run_id": args.run_id, "pilot_only": args.pilot, "created_at_utc": datetime.now(timezone.utc).isoformat(),
              "evaluation_partition": "validation", "final_test_evaluated": False, "config": config,
              "dataset_manifest_sha256": sha256(processed / "manifest.json"), "config_sha256": sha256(config_path),
              "source_code_hashes": source_hashes(), "training_compounds": len(train), "validation_compounds": len(validation),
              "training_hits": int(train_y.sum()), "validation_hits": int(validation_y.sum()),
              "fingerprint_array_mib": (train_x.nbytes + validation_x.nbytes) / 1024**2,
              "fingerprint_seconds": time.perf_counter() - start, "models": {}}
    save_json(destination / "run_started.json", result)
    nn_start = time.perf_counter()
    scores, indices, similarities, used = neighbour_predict(train_fps, validation_fps, train_y, config["nearest_neighbours"])
    nn = {"seconds": time.perf_counter() - nn_start, "runtime": runtime()}
    if not args.pilot:
        nn["metrics"] = metrics(validation_y, scores, keys, config["top_k"])
        nn["nearest_similarity_quantiles"] = {str(q): float(np.quantile(similarities, q)) for q in (0, 0.25, 0.5, 0.75, 1)}
        write_predictions(destination / "nearest_neighbour_validation.csv", validation, validation_y, scores,
                          {"nearest_train_key": [train[int(i)]["compound_key"] for i in indices],
                           "nearest_train_similarity": similarities, "neighbours_including_ties": used})
    result["models"]["nearest_neighbour"] = nn
    seeds = config["training_seeds"][:1] if args.pilot else config["training_seeds"]
    for seed in seeds:
        parameters = dict(config["rf"])
        if args.pilot:
            parameters["n_estimators"] = 50
        model = RandomForestClassifier(**parameters, random_state=seed, n_jobs=config["max_workers"])
        fit_start = time.perf_counter()
        with threadpool_limits(limits=config["max_workers"]):
            model.fit(train_x, train_y)
            fit_seconds = time.perf_counter() - fit_start
            predict_start = time.perf_counter()
            prediction = model.predict_proba(validation_x)[:, 1]
        info = {"fit_seconds": fit_seconds, "predict_seconds": time.perf_counter() - predict_start,
                "parameters": model.get_params(), "runtime": runtime()}
        if not args.pilot:
            info["metrics"] = metrics(validation_y, prediction, keys, config["top_k"])
            # Similarity bins are fixed independently of scores and are descriptive, not tuning criteria.
            info["distance_subgroups"] = {}
            for label, low, high in (("less_than_0.3", 0, 0.3), ("0.3_to_0.5", 0.3, 0.5), ("at_least_0.5", 0.5, 1.00001)):
                mask = (similarities >= low) & (similarities < high)
                info["distance_subgroups"][label] = {"n": int(mask.sum()), "hits": int(validation_y[mask].sum())}
                if mask.sum() and len(set(validation_y[mask])) == 2:
                    info["distance_subgroups"][label]["average_precision"] = float(average_precision_score(validation_y[mask], prediction[mask]))
            write_predictions(destination / f"random_forest_seed{seed}_validation.csv", validation, validation_y, prediction,
                              {"nearest_train_similarity": similarities})
            joblib.dump(model, destination / f"random_forest_seed{seed}.joblib", compress=3)
        result["models"][f"random_forest_seed{seed}"] = info
        save_json(destination / "results.json", {**result, "complete": False})
        print(f"RF seed {seed}: fit {fit_seconds:.1f}s; peak RSS {info['runtime']['peak_rss_mib']:.0f} MiB", flush=True)
    result.update({"complete": True, "total_seconds": time.perf_counter() - start, "runtime": runtime()})
    save_json(destination / "results.json", result)
    print(json.dumps({"run_id": args.run_id, "seconds": result["total_seconds"], "peak_rss_mib": result["runtime"]["peak_rss_mib"],
                      "final_test_evaluated": False, "metrics": {k: v.get("metrics") for k, v in result["models"].items()}}, indent=2))


if __name__ == "__main__":
    main()
