"""Resource-test and cache frozen CheMeleon embeddings; no activity labels loaded."""
import os
for variable in ("OMP_NUM_THREADS", "OPENBLAS_NUM_THREADS", "MKL_NUM_THREADS", "VECLIB_MAXIMUM_THREADS"):
    os.environ.setdefault(variable, "2")

import argparse
import hashlib
import importlib.metadata
import json
import time

import numpy as np
import pyarrow.parquet as pq
import torch
from chemprop import data, featurizers, nn
from torch.utils.data import DataLoader

from .common import ROOT, runtime, save_json, sha256, source_hashes


def parameter_digest(module):
    digest = hashlib.sha256()
    for key, value in sorted(module.state_dict().items()):
        digest.update(key.encode())
        digest.update(value.detach().cpu().numpy().tobytes())
    return digest.hexdigest()


def encode_rows(rows, encoder, aggregation, batch_size):
    points = [data.MoleculeDatapoint.from_smi(r["parent_smiles"]) for r in rows]
    dataset = data.MoleculeDataset(points, featurizers.SimpleMoleculeMolGraphFeaturizer())
    dataset.cache = False
    loader = DataLoader(dataset, batch_size=batch_size, shuffle=False, num_workers=0,
                        collate_fn=data.collate_batch, drop_last=False)
    with torch.inference_mode():
        for batch in loader:
            graph = batch[0]
            encoded = aggregation(encoder(graph), graph.batch).cpu().numpy()
            if not np.isfinite(encoded).all() or encoded.ndim != 2:
                raise ValueError("Non-finite or malformed encoder output")
            yield encoded


def main():
    parser = argparse.ArgumentParser()
    parser.add_argument("--pilot", action="store_true")
    parser.add_argument("--run-id", required=True)
    args = parser.parse_args()
    if not args.run_id or any(c not in "abcdefghijklmnopqrstuvwxyzABCDEFGHIJKLMNOPQRSTUVWXYZ0123456789-_" for c in args.run_id):
        raise ValueError("Invalid run id")
    config_path = ROOT / "configs/chemeleon_v0.json"
    config = json.loads(config_path.read_text())
    base = json.loads((ROOT / "configs/v0.json").read_text())
    if config["final_test_evaluation"] is not False or base["final_test_evaluation"] is not False:
        raise ValueError("Development-only encoder")
    if (config["frozen"], config["aggregation"], config["accelerator"], config["embedding_dtype"]) != (True, "mean", "cpu", "float32"):
        raise ValueError("Unsupported encoder configuration")
    torch.set_num_threads(config["cpu_threads"])
    torch.set_num_interop_threads(1)
    torch.use_deterministic_algorithms(True)
    start = time.perf_counter()
    destination = ROOT / "outputs" / args.run_id
    destination.mkdir(parents=True, exist_ok=False)
    source = ROOT / "data/processed/wong_v0"
    manifest = json.loads((source / "manifest.json").read_text())
    if manifest["config"] != base:
        raise ValueError("Data configuration differs from the frozen manifest")
    if not args.pilot:
        audit = json.loads((ROOT / config["overlap_report"]).read_text())
        if not audit["complete"] or audit["identity"]["pilot"] or audit["processed"] != 944296:
            raise ValueError("Complete pretraining structure audit required")
        if audit["identity"]["assignments_sha256"] != manifest["output_hashes"]["split_assignments.jsonl"]:
            raise ValueError("Audit belongs to another data split")
    rows = {}
    for partition in ("train", "validation"):
        path = source / (partition + ".parquet")
        if sha256(path) != manifest["output_hashes"][path.name]:
            raise ValueError("Data changed")
        rows[partition] = pq.read_table(path, columns=["compound_key", "parent_smiles", "heavy_atoms"]).to_pylist()
    full_sizes = {p: len(v) for p, v in rows.items()}
    largest = sorted(rows["train"], key=lambda r: (-r["heavy_atoms"], r["compound_key"]))[:config["batch_size"]]
    if args.pilot:
        rng = np.random.default_rng(config["pilot_sample_seed"])
        for p in rows:
            selected = rng.choice(len(rows[p]), config[f"pilot_{p}_n"], replace=False)
            rows[p] = [rows[p][int(i)] for i in selected]
    path = ROOT / "data/raw/chemeleon/chemeleon_mp.pt"
    if sha256(path) != config["checkpoint_sha256"]:
        raise ValueError("Checkpoint content changed")
    saved = torch.load(path, map_location="cpu", weights_only=True)
    encoder = nn.BondMessagePassing(**saved["hyper_parameters"])
    encoder.load_state_dict(saved["state_dict"], strict=True)
    encoder.requires_grad_(False).eval()
    aggregation = nn.MeanAggregation().eval()
    before = parameter_digest(encoder)
    result = {"run_id": args.run_id, "pilot_only": args.pilot, "config": config,
              "config_sha256": sha256(config_path), "dataset_manifest_sha256": sha256(source / "manifest.json"),
              "source_code_hashes": source_hashes(), "full_sizes": full_sizes, "embedding_dim": encoder.output_dim,
              "weights_sha256": config["checkpoint_sha256"], "parameter_count": sum(p.numel() for p in encoder.parameters()),
              "activity_labels_read": False, "final_test_evaluated": False, "partitions": {},
              "packages": {p: importlib.metadata.version(p) for p in ("torch", "chemprop", "rdkit", "numpy")}}
    save_json(destination / "run_started.json", result)
    if args.pilot:
        stress_start = time.perf_counter()
        stress = np.concatenate(list(encode_rows(largest, encoder, aggregation, config["batch_size"])))
        # Embeddings must be independent of batch composition and row order.
        singleton = np.concatenate(list(encode_rows(largest[:1], encoder, aggregation, 1)))
        np.testing.assert_allclose(stress[:1], singleton, rtol=1e-4, atol=1e-5)
        result["stress_batch"] = {"max_heavy_atoms": largest[0]["heavy_atoms"], "batch_size": len(largest),
                                  "seconds": time.perf_counter() - stress_start,
                                  "peak_rss_mib": runtime()["peak_rss_mib"], "batch_invariance_checked": True}
    for partition, molecules in rows.items():
        encode_start = time.perf_counter()
        cache_path = destination / f"{partition}_embeddings.npy"
        features = np.lib.format.open_memmap(cache_path, mode="w+", dtype=np.float32, shape=(len(molecules), encoder.output_dim))
        offset = 0
        for values in encode_rows(molecules, encoder, aggregation, config["batch_size"]):
            features[offset:offset + len(values)] = values
            offset += len(values)
            if runtime()["peak_rss_mib"] > config["process_memory_budget_mib"]:
                raise RuntimeError("Frozen-encoder process exceeded local memory budget")
            if offset % 1024 == 0:
                print(f"{partition}: {offset}/{len(molecules)}; {time.perf_counter()-encode_start:.1f}s", flush=True)
        if offset != len(molecules):
            raise ValueError("Encoder dropped molecules")
        features.flush()
        del features
        ids_path = destination / f"{partition}_compound_keys.json"
        save_json(ids_path, [r["compound_key"] for r in molecules])
        result["partitions"][partition] = {"n": offset, "seconds": time.perf_counter()-encode_start,
                                           "cache_sha256": sha256(cache_path), "ids_sha256": sha256(ids_path)}
    after = parameter_digest(encoder)
    if before != after or any(p.requires_grad for p in encoder.parameters()):
        raise ValueError("Frozen encoder parameters changed")
    result["encoder_parameters_unchanged"] = True
    if args.pilot:
        seconds = sum(p["seconds"] for p in result["partitions"].values())
        n = sum(p["n"] for p in result["partitions"].values())
        result["estimated_full_embedding_minutes"] = seconds / n * sum(full_sizes.values()) / 60
        result["full_embedding_cache_mib"] = sum(full_sizes.values()) * encoder.output_dim * 4 / 1024**2
    else:
        result["overlap_report_sha256"] = sha256(ROOT / config["overlap_report"])
    result.update({"complete": True, "total_seconds": time.perf_counter()-start, "runtime": runtime()})
    save_json(destination / "results.json", result)
    print(json.dumps({k: result[k] for k in ("run_id", "complete", "total_seconds", "runtime")}), flush=True)


if __name__ == "__main__":
    main()
