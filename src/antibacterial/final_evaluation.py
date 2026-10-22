"""Freeze a V0 evaluation, reproduce validation, then score the held-out test once."""
import os
for variable in ("OMP_NUM_THREADS", "OPENBLAS_NUM_THREADS", "MKL_NUM_THREADS", "VECLIB_MAXIMUM_THREADS"):
    os.environ.setdefault(variable, "2")

import argparse
import gc
import json
import time
from datetime import datetime, timezone

import joblib
import numpy as np
import pyarrow.parquet as pq
import torch
from chemprop import data, featurizers, nn
from sklearn.metrics import average_precision_score
from threadpoolctl import threadpool_limits

from .baseline import fingerprints, metrics, neighbour_predict, write_predictions
from .common import ROOT, runtime, save_json, sha256, source_hashes
from .compare import verify_predictions
from .frozen_encoder import encode_rows, parameter_digest
from .neural import make_loader, make_model, machine_runtime

PROTOCOL = ROOT / "configs/final_evaluation_v1.json"
SMOKE = ROOT / "outputs/final-validation-smoke-v1"
DESTINATION = ROOT / "outputs/final-test-v1"


def build_specification():
    """Uses manifests and completed development runs, never individual test rows."""
    base = json.loads((ROOT / "configs/v0.json").read_text())
    manifest_path = ROOT / "data/processed/wong_v0/manifest.json"
    manifest = json.loads(manifest_path.read_text())
    paths = [manifest_path, ROOT / "configs/v0.json", ROOT / "configs/chemprop_v0.json",
             ROOT / "configs/chemeleon_v0.json", ROOT / "requirements-neural.lock.txt",
             ROOT / "reports/pretrained_comparison.json", ROOT / "reports/chemeleon-audit-full-v1.json",
             ROOT / "data/raw/chemeleon/chemeleon_mp.pt"]
    paths += [manifest_path.parent / p for p in manifest["output_hashes"]]
    for name, digest in manifest["output_hashes"].items():
        if sha256(manifest_path.parent / name) != digest:
            raise ValueError(f"Frozen data changed: {name}")
    entries = []
    baseline_dir = ROOT / "outputs/development-20260909-numeric-fix"
    frozen_dir = ROOT / "outputs/chemeleon-rf-v1"
    for directory in (baseline_dir, frozen_dir):
        run = json.loads((directory / "results.json").read_text())
        if not run["complete"] or run["pilot_only"] or run["dataset_manifest_sha256"] != sha256(manifest_path):
            raise ValueError("Incomplete or mismatched development run")
        paths.append(directory / "results.json")
    entries.append({"name": "nearest_neighbour", "family": "nearest_neighbour", "seed": None,
                    "validation_predictions": str((baseline_dir / "nearest_neighbour_validation.csv").relative_to(ROOT))})
    for family, directory, prefix in (("random_forest", baseline_dir, "random_forest"),
                                       ("chemeleon_frozen_rf", frozen_dir, "frozen_rf")):
        for seed in base["training_seeds"]:
            entries.append({"name": f"{family}_seed{seed}", "family": family, "seed": seed,
                            "model": str((directory / f"{prefix}_seed{seed}.joblib").relative_to(ROOT)),
                            "validation_predictions": str((directory / f"{prefix}_seed{seed}_validation.csv").relative_to(ROOT))})
    for seed in base["training_seeds"]:
        directory = ROOT / f"outputs/chemprop-seed{seed}-20260909"
        run = json.loads((directory / "results.json").read_text())
        if not run["complete"] or run["pilot_only"] or run["dataset_manifest_sha256"] != sha256(manifest_path):
            raise ValueError("Incomplete or mismatched neural run")
        if sha256(ROOT / run["checkpoint_path"]) != run["checkpoint_sha256"]:
            raise ValueError("Neural checkpoint changed")
        paths.append(directory / "results.json")
        entries.append({"name": f"chemprop_scratch_seed{seed}", "family": "chemprop_scratch", "seed": seed,
                        "model": run["checkpoint_path"], "selected_epoch": run["selected_epoch"],
                        "validation_predictions": str((directory / "chemprop_validation.csv").relative_to(ROOT))})
    for entry in entries:
        paths.append(ROOT / entry["validation_predictions"])
        if "model" in entry:
            paths.append(ROOT / entry["model"])
    return {"version": "V0-final-v1", "models": entries, "primary_metric": "average_precision",
            "top_k": base["top_k"], "primary_practical_k": 50,
            "validation_selected_family": "chemeleon_frozen_rf", "reference_family": "nearest_neighbour",
            "model_selection": "All predefined seeds; family mean AP. No best-test-seed selection or ensemble.",
            "training_policy": "Use saved train-only models; no refitting, tuning, calibration or test-based changes.",
            "tie_policy": "Descending score, ascending compound key; also report cutoff-tie-averaged precision.",
            "subgroups": {"similarity_bins": [0.3, 0.5], "pretraining_identity": "Confirmed full parent InChIKey match vs no confirmed match; descriptive only."},
            "uncertainty_policy": "Show all seeds and sample SD, not confidence intervals or significance claims; one fixed split.",
            "future_use": "After scoring, this test is a historical V0 benchmark, not an untouched holdout for adaptive V1 development.",
            "source_hashes": source_hashes(),
            "input_hashes": {str(p.relative_to(ROOT)): sha256(p) for p in sorted(set(paths))},
            "packages": machine_runtime()["packages"]}


def verify_pinned(spec):
    for path, digest in {**spec["source_hashes"], **spec["input_hashes"]}.items():
        if sha256(ROOT / path) != digest:
            raise ValueError(f"Pinned input or source changed: {path}")
    if machine_runtime()["packages"] != spec["packages"]:
        raise ValueError("Environment differs from validation inference check")


def predict_all(spec, partition, destination):
    """Predict from structures only; labels for queries are loaded later for scoring."""
    base = json.loads((ROOT / "configs/v0.json").read_text())
    processed = ROOT / "data/processed/wong_v0"
    train = pq.read_table(processed / "train.parquet").to_pylist()
    query = pq.read_table(processed / f"{partition}.parquet", columns=["compound_key", "parent_smiles"]).to_pylist()
    train_fps, train_x = fingerprints(train, base["fingerprint"])
    del train_x
    query_fps, query_x = fingerprints(query, base["fingerprint"])
    scores, _, similarity, _ = neighbour_predict(train_fps, query_fps, np.asarray([r["activity"] for r in train]), base["nearest_neighbours"])
    predictions = {"nearest_neighbour": scores}
    del train_fps, query_fps, train
    embedding = None
    encoder_config = json.loads((ROOT / "configs/chemeleon_v0.json").read_text())
    neural_config = json.loads((ROOT / "configs/chemprop_v0.json").read_text())
    timings = {}
    for entry in spec["models"][1:]:
        start = time.perf_counter()
        family = entry["family"]
        if family == "chemeleon_frozen_rf" and embedding is None:
            saved = torch.load(ROOT / "data/raw/chemeleon/chemeleon_mp.pt", map_location="cpu", weights_only=True)
            encoder = nn.BondMessagePassing(**saved["hyper_parameters"])
            encoder.load_state_dict(saved["state_dict"], strict=True)
            encoder.requires_grad_(False).eval()
            before = parameter_digest(encoder)
            embedding = np.concatenate(list(encode_rows(query, encoder, nn.MeanAggregation().eval(), encoder_config["batch_size"])))
            if before != parameter_digest(encoder):
                raise ValueError("Encoder changed during prediction")
            del encoder, saved
        if family == "chemprop_scratch":
            model = make_model(neural_config)
            # Checkpoint was produced locally and is pinned in this protocol.
            saved = torch.load(ROOT / entry["model"], map_location="cpu", weights_only=False)
            model.load_state_dict(saved["state_dict"], strict=True)
            model.requires_grad_(False).eval()
            points = [data.MoleculeDatapoint.from_smi(r["parent_smiles"]) for r in query]
            dataset = data.MoleculeDataset(points, featurizers.SimpleMoleculeMolGraphFeaturizer())
            dataset.cache = False
            with torch.inference_mode():
                values = [model(batch[0], batch[1], batch[2]).cpu().numpy().reshape(-1)
                          for batch in make_loader(dataset, neural_config["batch_size"])]
            score = np.concatenate(values)
            del saved, points, dataset
        else:
            model = joblib.load(ROOT / entry["model"])
            with threadpool_limits(limits=2):
                score = model.predict_proba(embedding if family == "chemeleon_frozen_rf" else query_x)[:, 1]
        if len(score) != len(query) or not np.isfinite(score).all() or np.any((score < 0) | (score > 1)):
            raise ValueError("Incomplete or invalid predictions")
        predictions[entry["name"]] = score
        timings[entry["name"]] = time.perf_counter() - start
        del model
        gc.collect()
        if runtime()["peak_rss_mib"] > 3072:
            raise RuntimeError("Evaluation exceeded the local memory budget")
        print(f"Predicted {entry['name']}: {timings[entry['name']]:.1f}s", flush=True)
    matrix = np.column_stack([predictions[e["name"]] for e in spec["models"]])
    np.save(destination / "unscored_predictions.npy", matrix, allow_pickle=False)
    save_json(destination / "prediction_ids.json", [r["compound_key"] for r in query])
    save_json(destination / "predictions_sealed.json", {"query_activity_labels_read": False,
              "scores_sha256": sha256(destination / "unscored_predictions.npy"),
              "ids_sha256": sha256(destination / "prediction_ids.json"), "timings": timings,
              "model_order": [e["name"] for e in spec["models"]]})
    return predictions, similarity


def score_and_verify(spec, partition, destination, predictions, similarity):
    rows = pq.read_table(ROOT / f"data/processed/wong_v0/{partition}.parquet").to_pylist()
    y = np.asarray([r["activity"] for r in rows])
    ids = [r["compound_key"] for r in rows]
    if ids != json.loads((destination / "prediction_ids.json").read_text()):
        raise ValueError("Prediction-to-label order changed")
    audit = json.loads((ROOT / "reports/chemeleon-audit-full-v1.json").read_text())
    exposed_keys = set(audit["overlaps"][partition]["exact_compound_keys"])
    exposed = np.asarray([key in exposed_keys for key in ids])
    masks = {"similarity_less_than_0.3": similarity < .3,
             "similarity_0.3_to_0.5": (similarity >= .3) & (similarity < .5), "similarity_at_least_0.5": similarity >= .5,
             "pretraining_identity_exposed": exposed, "no_confirmed_pretraining_identity_match": ~exposed}
    records = []
    for entry in spec["models"]:
        scores = predictions[entry["name"]]
        measured = metrics(y, scores, ids, spec["top_k"])
        path = destination / f"{entry['name']}_{partition}.csv"
        write_predictions(path, rows, y, scores, {"nearest_train_similarity": similarity})
        _, recomputed, _ = verify_predictions(path, rows, spec["top_k"])
        if measured != recomputed:
            raise ValueError("Saved predictions do not reproduce reported metrics")
        if partition == "validation":
            original, original_metrics, _ = verify_predictions(ROOT / entry["validation_predictions"], rows, spec["top_k"])
            np.testing.assert_allclose(scores, original, rtol=0, atol=1e-7)
            if abs(measured["average_precision"] - original_metrics["average_precision"]) > 1e-12 or measured["top_k"] != original_metrics["top_k"]:
                raise ValueError("New inference code does not reproduce development metrics")
        groups = {name: {"n": int(mask.sum()), "hits": int(y[mask].sum()),
                        "average_precision": float(average_precision_score(y[mask], scores[mask])) if len(set(y[mask])) == 2 else None}
                  for name, mask in masks.items()}
        records.append({**entry, "metrics": measured, "subgroups": groups, "predictions_sha256": sha256(path)})
    aggregates = {}
    for family in dict.fromkeys(e["family"] for e in spec["models"]):
        selected = [r for r in records if r["family"] == family]
        ap = np.asarray([r["metrics"]["average_precision"] for r in selected])
        aggregates[family] = {"n_runs": len(selected), "ap_mean": float(ap.mean()),
                              "ap_sample_std": float(ap.std(ddof=1)) if len(ap) > 1 else None,
                              "top50_hits": [r["metrics"]["top_k"]["50"]["hits"] for r in selected]}
    return {"evaluation_partition": partition, "n": len(rows), "hits": int(y.sum()),
            "all_predictions_verified": True, "aggregates": aggregates, "runs": records}


def main():
    parser = argparse.ArgumentParser()
    parser.add_argument("action", choices=["smoke", "freeze", "test"])
    args = parser.parse_args()
    torch.set_num_threads(2)
    torch.set_num_interop_threads(1)
    torch.use_deterministic_algorithms(True)
    start = time.perf_counter()
    if args.action in ("smoke", "freeze"):
        spec = build_specification()
        if args.action == "freeze":
            smoke = json.loads((SMOKE / "results.json").read_text())
            if not smoke["complete"] or smoke["specification"] != spec:
                raise ValueError("Run validation smoke check with the current exact specification first")
            if PROTOCOL.exists():
                raise FileExistsError("Protocol already frozen; do not overwrite")
            save_json(PROTOCOL, {"frozen_at_utc": datetime.now(timezone.utc).isoformat(),
                      "validation_smoke_sha256": sha256(SMOKE / "results.json"), "specification": spec})
            print(json.dumps({"frozen_protocol": str(PROTOCOL), "sha256": sha256(PROTOCOL)}))
            return
        partition, destination = "validation", SMOKE
    else:
        protocol = json.loads(PROTOCOL.read_text())
        spec = protocol["specification"]
        if sha256(SMOKE / "results.json") != protocol["validation_smoke_sha256"]:
            raise ValueError("Validation smoke record changed")
        partition, destination = "test", DESTINATION
    verify_pinned(spec)
    destination.mkdir(parents=True, exist_ok=False)
    save_json(destination / "run_started.json", {"started_at_utc": datetime.now(timezone.utc).isoformat(),
              "partition": partition, "specification": spec,
              "protocol_sha256": sha256(PROTOCOL) if args.action == "test" else None})
    predictions, similarity = predict_all(spec, partition, destination)
    report = score_and_verify(spec, partition, destination, predictions, similarity)
    verify_pinned(spec)
    report.update({"complete": True, "specification": spec, "total_seconds": time.perf_counter() - start,
                   "runtime": machine_runtime(), "final_test_evaluated": partition == "test",
                   "protocol_sha256": sha256(PROTOCOL) if partition == "test" else None})
    save_json(destination / "results.json", report)
    if partition == "test":
        save_json(ROOT / "reports/final_test_comparison.json", report)
    print(json.dumps({"partition": partition, "seconds": report["total_seconds"], "aggregates": report["aggregates"]}, indent=2))


if __name__ == "__main__":
    main()
