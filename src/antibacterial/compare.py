"""Independently reconcile validation outputs and summarize the completed runs."""
import argparse
import csv
import json

import numpy as np
import pyarrow.parquet as pq
from sklearn.metrics import average_precision_score

from .baseline import metrics
from .common import ROOT, save_json, sha256


def verify_predictions(path, validation, top_k):
    with path.open() as handle:
        predicted = list(csv.DictReader(handle))
    expected = {r["compound_key"]: r for r in validation}
    if len(predicted) != len(expected) or len({r["compound_key"] for r in predicted}) != len(expected):
        raise ValueError(f"Incorrect or duplicated prediction coverage: {path}")
    by_key = {r["compound_key"]: r for r in predicted}
    if set(by_key) != set(expected):
        raise ValueError(f"Incorrect prediction IDs: {path}")
    for key, source in expected.items():
        row = by_key[key]
        if int(row["observed_label"]) != source["activity"] or row["parent_smiles"] != source["parent_smiles"]:
            raise ValueError(f"Prediction-to-source mismatch: {key}")
        if json.loads(row["source_rows"]) != source["source_rows"]:
            raise ValueError(f"Prediction provenance mismatch: {key}")
    scores = np.asarray([float(by_key[r["compound_key"]]["predicted_score"]) for r in validation])
    if not np.isfinite(scores).all() or np.any((scores < 0) | (scores > 1)):
        raise ValueError(f"Invalid scores: {path}")
    labels, keys = [r["activity"] for r in validation], [r["compound_key"] for r in validation]
    return scores, metrics(labels, scores, keys, top_k), by_key


def main():
    parser = argparse.ArgumentParser()
    parser.add_argument("--baseline-run", required=True)
    parser.add_argument("--neural-runs", nargs="+", required=True)
    parser.add_argument("--frozen-runs", nargs="*", default=[])
    parser.add_argument("--report", default="reports/chemprop_comparison.json")
    args = parser.parse_args()
    processed = ROOT / "data/processed/wong_v0"
    manifest_path = processed / "manifest.json"
    manifest = json.loads(manifest_path.read_text())
    if sha256(processed / "validation.parquet") != manifest["output_hashes"]["validation.parquet"]:
        raise ValueError("Validation data changed")
    validation = pq.read_table(processed / "validation.parquet").to_pylist()
    reference_hash = sha256(manifest_path)
    baseline_dir = ROOT / "outputs" / args.baseline_run
    baseline = json.loads((baseline_dir / "results.json").read_text())
    records = []
    paths = [(baseline_dir / "nearest_neighbour_validation.csv", baseline, "nearest_neighbour", None,
              baseline["models"]["nearest_neighbour"]["metrics"])]
    for seed in manifest["config"]["training_seeds"]:
        paths.append((baseline_dir / f"random_forest_seed{seed}_validation.csv", baseline, "random_forest", seed,
                      baseline["models"][f"random_forest_seed{seed}"]["metrics"]))
    neural = []
    for name in args.neural_runs:
        directory = ROOT / "outputs" / name
        run = json.loads((directory / "results.json").read_text())
        neural.append(run)
        paths.append((directory / "chemprop_validation.csv", run, "chemprop_scratch", run["seed"], run["metrics"]))
    if len({r["seed"] for r in neural}) != len(neural) or len({r["config_sha256"] for r in neural}) != 1:
        raise ValueError("Neural seeds must be distinct and use the same configuration")
    if len(args.frozen_runs) > 1:
        raise ValueError("Compare one declared frozen-encoder configuration per report")
    exposed_validation_keys = None
    overlap_report_hash = None
    for name in args.frozen_runs:
        directory = ROOT / "outputs" / name
        run = json.loads((directory / "results.json").read_text())
        if run["family"] != "chemeleon_frozen_rf":
            raise ValueError("Unexpected frozen model family")
        overlap_path = ROOT / run["encoder_config"]["overlap_report"]
        overlap = json.loads(overlap_path.read_text())
        embedding_path = ROOT / "outputs" / run["embedding_run"] / "results.json"
        if sha256(embedding_path) != run["embedding_results_sha256"]:
            raise ValueError("Embedding provenance changed")
        embedding = json.loads(embedding_path.read_text())
        if sha256(overlap_path) != embedding["overlap_report_sha256"]:
            raise ValueError("Structural exposure audit changed")
        if not overlap["complete"] or overlap["identity"]["pilot"]:
            raise ValueError("Incomplete structural exposure audit")
        if overlap["identity"]["assignments_sha256"] != manifest["output_hashes"]["split_assignments.jsonl"]:
            raise ValueError("Structural exposure audit uses a different split")
        exposed_validation_keys = set(overlap["overlaps"]["validation"]["exact_compound_keys"])
        if not exposed_validation_keys.issubset({r["compound_key"] for r in validation}):
            raise ValueError("Exposure audit contains unexpected validation compounds")
        overlap_report_hash = sha256(overlap_path)
        for seed in manifest["config"]["training_seeds"]:
            paths.append((directory / f"frozen_rf_seed{seed}_validation.csv", run, "chemeleon_frozen_rf", seed,
                          run["models"][f"frozen_rf_seed{seed}"]["metrics"]))
    similarities = None
    y = np.asarray([r["activity"] for r in validation])
    for path, run, family, seed, reported in paths:
        if not run["complete"] or run["pilot_only"] or run["final_test_evaluated"] or run["evaluation_partition"] != "validation":
            raise ValueError("Only complete development runs can be compared")
        if run["dataset_manifest_sha256"] != reference_hash:
            raise ValueError("Models were trained on different data versions")
        scores, measured, by_key = verify_predictions(path, validation, manifest["config"]["top_k"])
        if not np.isclose(measured["average_precision"], reported["average_precision"], atol=1e-12, rtol=0):
            raise ValueError("Recomputed AP differs")
        if measured["top_k"] != reported["top_k"]:
            raise ValueError("Recomputed top-K differs")
        if family == "nearest_neighbour":
            similarities = np.asarray([float(by_key[r["compound_key"]]["nearest_train_similarity"]) for r in validation])
        groups = {}
        for label, low, high in (("less_than_0.3", 0, 0.3), ("0.3_to_0.5", 0.3, 0.5), ("at_least_0.5", 0.5, 1.00001)):
            mask = (similarities >= low) & (similarities < high)
            groups[label] = {"n": int(mask.sum()), "hits": int(y[mask].sum()),
                             "average_precision": float(average_precision_score(y[mask], scores[mask])) if len(set(y[mask])) == 2 else None}
        records.append({"family": family, "seed": seed, "run_id": run["run_id"], "metrics": measured,
                        "distance_subgroups": groups, "predictions_sha256": sha256(path)})
        if exposed_validation_keys is not None:
            exposed = np.asarray([r["compound_key"] in exposed_validation_keys for r in validation])
            records[-1]["pretraining_identity_subgroups"] = {}
            for label, mask in (("identity_exposed", exposed), ("no_confirmed_identity_match", ~exposed)):
                records[-1]["pretraining_identity_subgroups"][label] = {
                    "n": int(mask.sum()), "hits": int(y[mask].sum()),
                    "average_precision": float(average_precision_score(y[mask], scores[mask])) if len(set(y[mask])) == 2 else None}
    aggregates = {}
    for family in dict.fromkeys(r["family"] for r in records):
        values = [r for r in records if r["family"] == family]
        aps = np.asarray([r["metrics"]["average_precision"] for r in values])
        aggregates[family] = {"n_runs": len(values), "ap_mean": float(aps.mean()),
                              "ap_sample_std": float(aps.std(ddof=1)) if len(values) > 1 else None,
                              "ap_min": float(aps.min()), "ap_max": float(aps.max()),
                              "top50_hits": [r["metrics"]["top_k"]["50"]["hits"] for r in values]}
    report = {"evaluation_partition": "validation", "final_test_evaluated": False,
              "dataset_manifest_sha256": reference_hash, "validation_compounds": len(validation),
              "validation_hits": int(y.sum()), "all_predictions_verified": True,
              "all_planned_neural_seeds_complete": sorted(r["seed"] for r in neural) == sorted(neural[0]["config"]["training_seeds"]),
              "aggregates": aggregates, "runs": records,
              "pretraining_overlap_report_sha256": overlap_report_hash,
              "limitations": ["Validation was used for neural checkpoint selection; these are development comparisons, not final-test estimates.",
                              "Seed variation measures training randomness on one split, not cross-dataset uncertainty.",
                              "The predefined nearest-neighbour similarity bins are descriptive and have few positive examples.",
                              "The frozen encoder comparison changes the representation; it does not isolate pretraining from architecture and model size.",
                              "No confirmed pretraining identity match does not establish complete structural non-exposure; related structures and unresolved cases can remain.",
                              "No experimental discovery, natural-product-domain or actinomycete-origin validation has been performed."]}
    save_json(ROOT / args.report, report)
    print(json.dumps(report["aggregates"], indent=2))


if __name__ == "__main__":
    main()
