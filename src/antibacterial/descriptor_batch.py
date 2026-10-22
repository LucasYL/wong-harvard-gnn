"""Freeze and run a sequential, restartable descriptor ensemble experiment."""
import argparse
import fcntl
import json
import os
import subprocess
import sys
import time
from datetime import datetime, timezone

import numpy as np

from .common import ROOT, save_json, sha256, source_hashes
from .compare import verify_predictions
from .descriptors import CONFIG, CACHE, read_rows, load_features
from .neural_descriptors import all_metrics, mean_scores

PROTOCOL = ROOT / "configs/descriptor_experiment_frozen_v1.json"
BATCH = ROOT / "outputs/chemprop-descriptors-v1"


def experiment_identity():
    return {"config_sha256": sha256(CONFIG), "descriptor_manifest_sha256": sha256(CACHE / "manifest.json"),
            "dataset_manifest_sha256": sha256(ROOT / "data/processed/wong_v0/manifest.json"),
            "source_code_hashes": source_hashes()}


def atomic_json(path, value):
    temp = path.with_name(path.name + ".tmp")
    save_json(temp, value)
    os.replace(temp, path)


def freeze():
    if PROTOCOL.exists():
        raise ValueError("Frozen protocol cannot be overwritten")
    for partition in ("train", "validation"):
        load_features(partition, read_rows(partition))
    config = json.loads(CONFIG.read_text())
    pilot = json.loads((ROOT / "outputs/chemprop-descriptors-pilot-v1/results.json").read_text())
    if not pilot["complete"] or not pilot["pilot_only"] or pilot["config_sha256"] != sha256(CONFIG):
        raise ValueError("A successful matching resource pilot is required")
    if pilot["source_code_hashes"] != source_hashes():
        raise ValueError("Run the pilot after the final code changes")
    if pilot["runtime"]["peak_rss_mib"] > config["process_memory_budget_mib"]:
        raise ValueError("Pilot exceeds memory budget")
    snapshots = ROOT / "reports/source_snapshots/descriptor-v1"
    original = experiment_identity()
    snapshots.mkdir(parents=True, exist_ok=False)
    for name in original["source_code_hashes"]:
        target = snapshots / name
        target.parent.mkdir(parents=True, exist_ok=True)
        target.write_bytes((ROOT / name).read_bytes())
    baseline_hashes = {}
    for name in config["baseline_runs"]:
        for suffix in ("results.json", "chemprop_validation.csv"):
            path = ROOT / "outputs" / name / suffix
            baseline_hashes[str(path.relative_to(ROOT))] = sha256(path)
    root = ROOT / "outputs" / config["baseline_fingerprint_run"]
    for path in [root / "results.json", root / "nearest_neighbour_validation.csv"] + list(root.glob("random_forest_seed*_validation.csv")):
        baseline_hashes[str(path.relative_to(ROOT))] = sha256(path)
    result = {"created_at_utc": datetime.now(timezone.utc).isoformat(), "identity": original,
              "config": config, "pilot_results_sha256": sha256(ROOT / "outputs/chemprop-descriptors-pilot-v1/results.json"),
              "baseline_hashes": baseline_hashes, "primary_metric": "average_precision",
              "secondary_metrics": ["trapezoidal_pr_auc", "top20", "top50", "top100"],
              "evaluation": "Validation development only; checkpoint and model choices are not independent estimates.",
              "no_final_test_reads_or_predictions": True}
    save_json(PROTOCOL, result)
    print(json.dumps({"frozen": str(PROTOCOL), "sha256": sha256(PROTOCOL)}), flush=True)


def check_protocol():
    protocol = json.loads(PROTOCOL.read_text())
    if protocol["identity"] != experiment_identity():
        raise ValueError("Frozen training identity changed")
    for name, digest in protocol["baseline_hashes"].items():
        if sha256(ROOT / name) != digest:
            raise ValueError("Baseline file changed")
    return protocol


def verify_run(run_id, validation, top_k):
    path = ROOT / "outputs" / run_id
    result = json.loads((path / "results.json").read_text())
    identity = experiment_identity()
    if not result["complete"] or result["pilot_only"] or result["final_test_evaluated"]:
        raise ValueError("Not a complete development run")
    if any(result[k] != v for k, v in identity.items()):
        raise ValueError("Run identity differs from experiment")
    if sha256(ROOT / result["checkpoint_path"]) != result["checkpoint_sha256"]:
        raise ValueError("Checkpoint changed")
    scores, measured, _ = verify_predictions(path / "chemprop_validation.csv", validation, top_k)
    if abs(measured["average_precision"] - result["metrics"]["average_precision"]) > 1e-12 or measured["top_k"] != result["metrics"]["top_k"]:
        raise ValueError("Saved metrics disagree with independently read predictions")
    return result, scores


def summarize():
    protocol = check_protocol()
    config = protocol["config"]
    validation = read_rows("validation", labels=True)
    top_k = json.loads((ROOT / "configs/v0.json").read_text())["top_k"]
    labels, keys = [r["activity"] for r in validation], [r["compound_key"] for r in validation]
    records, descriptor_scores, plain_scores = [], [], []
    for seed in config["training_seeds"]:
        run_id = f"chemprop-descriptors-v1-seed{seed}"
        path = ROOT / "outputs" / run_id / "results.json"
        if not path.exists():
            continue
        result, scores = verify_run(run_id, validation, top_k)
        if result["seed"] != seed:
            raise ValueError("Member seed mismatch")
        descriptor_scores.append((seed, scores))
        records.append({"name": run_id, "seed": seed, "kind": "descriptor_single",
                        "metrics": all_metrics(labels, scores, keys, top_k)})
    for run_id in config["baseline_runs"]:
        path = ROOT / "outputs" / run_id
        result = json.loads((path / "results.json").read_text())
        if result["dataset_manifest_sha256"] != protocol["identity"]["dataset_manifest_sha256"]:
            raise ValueError("Baseline dataset differs")
        scores, _, _ = verify_predictions(path / "chemprop_validation.csv", validation, top_k)
        plain_scores.append(scores)
        records.append({"name": run_id, "seed": result["seed"], "kind": "no_descriptor_single",
                        "metrics": all_metrics(labels, scores, keys, top_k)})
    ensembles = {"no_descriptors_3": (config["training_seeds"][:3], mean_scores(plain_scores))}
    by_seed = dict(descriptor_scores)
    for size in config["ensemble_sizes"]:
        selected = config["training_seeds"][:size]
        if all(seed in by_seed for seed in selected):
            ensembles[f"descriptors_{size}"] = (selected, mean_scores([by_seed[s] for s in selected]))
    from .baseline import write_predictions
    for name, (seeds, scores) in ensembles.items():
        write_predictions(BATCH / f"{name}_validation.csv", validation, labels, scores)
        records.append({"name": name, "kind": "equal_weight_prediction_ensemble", "seeds": seeds,
                        "metrics": all_metrics(labels, scores, keys, top_k)})
    root = ROOT / "outputs" / config["baseline_fingerprint_run"]
    for path in [root / "nearest_neighbour_validation.csv"] + sorted(root.glob("random_forest_seed*_validation.csv")):
        scores, _, _ = verify_predictions(path, validation, top_k)
        records.append({"name": path.stem, "kind": "fingerprint_baseline",
                        "metrics": all_metrics(labels, scores, keys, top_k)})
    report = {"complete": len(descriptor_scores) == len(config["training_seeds"]),
              "completed_descriptor_models": len(descriptor_scores), "planned_descriptor_models": len(config["training_seeds"]),
              "protocol_sha256": sha256(PROTOCOL), "evaluation_partition": "validation", "final_test_evaluated": False,
              "n": len(validation), "hits": sum(labels), "records": records,
              "limitations": ["Development results after earlier V0 test exposure; no new independent test claim.",
                              "Three-seed matched comparison isolates descriptor addition under the fixed light configuration.",
                              "Ten-member ensemble is an additional size comparison, not a ten-versus-ten descriptor ablation.",
                              "No new wet-lab, natural-product or actinomycete validation."]}
    atomic_json(BATCH / "comparison.json", report)
    atomic_json(ROOT / "reports/descriptor_ensemble_comparison.json", report)
    return report


def run(resume=False):
    protocol = check_protocol()
    BATCH.mkdir(parents=True, exist_ok=True)
    with (BATCH / "batch.lock").open("a") as lock:
        try:
            fcntl.flock(lock, fcntl.LOCK_EX | fcntl.LOCK_NB)
        except BlockingIOError as exc:
            raise ValueError("A descriptor batch is already running") from exc
        if (BATCH / "status.json").exists() and not resume:
            raise ValueError("Batch exists; use --resume to verify and continue")
        state = {"state": "running", "pid": os.getpid(), "started_at_utc": datetime.now(timezone.utc).isoformat(),
                 "protocol_sha256": sha256(PROTOCOL), "completed_seeds": [], "planned_seeds": protocol["config"]["training_seeds"]}
        atomic_json(BATCH / "status.json", state)
        try:
            for seed in state["planned_seeds"]:
                check_protocol()
                run_id = f"chemprop-descriptors-v1-seed{seed}"
                destination = ROOT / "outputs" / run_id
                if (destination / "results.json").exists():
                    if not resume:
                        raise ValueError("Existing member requires --resume")
                    summarize()
                    state["completed_seeds"].append(seed)
                    continue
                command = [sys.executable, "-m", "antibacterial.neural_descriptors", "--run-id", run_id, "--seed", str(seed)]
                if destination.exists():
                    if not resume:
                        raise ValueError("Incomplete member requires --resume")
                    command.append("--resume")
                state.update(current_seed=seed, updated_at_utc=datetime.now(timezone.utc).isoformat())
                atomic_json(BATCH / "status.json", state)
                print(json.dumps({"starting_seed": seed, "completed_seeds": state["completed_seeds"]}), flush=True)
                with (ROOT / "logs" / f"{run_id}.log").open("a") as log:
                    child = subprocess.run(command, cwd=ROOT, stdout=log, stderr=subprocess.STDOUT, check=False)
                if child.returncode:
                    raise RuntimeError(f"Seed {seed} exited with code {child.returncode}; inspect its log")
                summarize()
                state["completed_seeds"].append(seed)
                atomic_json(BATCH / "status.json", state)
            report = summarize()
            state.update(state="complete", current_seed=None, completed_at_utc=datetime.now(timezone.utc).isoformat())
            atomic_json(BATCH / "status.json", state)
            print(json.dumps({"complete": report["complete"], "report": "reports/descriptor_ensemble_comparison.json"}), flush=True)
        except BaseException as exc:
            state.update(state="failed", error=str(exc), updated_at_utc=datetime.now(timezone.utc).isoformat())
            atomic_json(BATCH / "status.json", state)
            raise


def main():
    parser = argparse.ArgumentParser()
    parser.add_argument("action", choices=("freeze", "run", "summarize", "status"))
    parser.add_argument("--resume", action="store_true")
    args = parser.parse_args()
    if args.action == "freeze":
        freeze()
    elif args.action == "run":
        run(args.resume)
    elif args.action == "summarize":
        print(json.dumps(summarize()), flush=True)
    else:
        state = json.loads((BATCH / "status.json").read_text())
        try:
            os.kill(state["pid"], 0)
            state["pid_currently_exists"] = True
        except ProcessLookupError:
            state["pid_currently_exists"] = False
        print(json.dumps(state), flush=True)


if __name__ == "__main__":
    main()
