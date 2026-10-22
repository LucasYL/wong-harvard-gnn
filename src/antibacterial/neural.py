"""Local Chemprop development runner. Never loads or evaluates final test rows."""
import os
for variable in ("OMP_NUM_THREADS", "OPENBLAS_NUM_THREADS", "MKL_NUM_THREADS", "VECLIB_MAXIMUM_THREADS"):
    os.environ.setdefault(variable, "2")

import argparse
import importlib.metadata
import json
import time
from datetime import datetime, timezone

import numpy as np
import pyarrow.parquet as pq
import torch
from chemprop import data, featurizers, models, nn
from lightning import pytorch as pl
from lightning.pytorch.callbacks import EarlyStopping, ModelCheckpoint
from lightning.pytorch.loggers import CSVLogger
from torch.utils.data import DataLoader

from .baseline import metrics, write_predictions
from .common import ROOT, runtime, save_json, sha256, source_hashes
from .neural_metrics import AveragePrecision


def machine_runtime():
    info = runtime()
    info["packages"].update({p: importlib.metadata.version(p) for p in ("torch", "chemprop", "lightning", "torchmetrics")})
    info.update({"torch_threads": torch.get_num_threads(), "mps_available": torch.backends.mps.is_available(),
                 "accelerator_used": "cpu"})
    return info


def make_dataset(rows):
    points = [data.MoleculeDatapoint.from_smi(r["parent_smiles"], np.asarray([r["activity"]], dtype=np.float32)) for r in rows]
    dataset = data.MoleculeDataset(points, featurizers.SimpleMoleculeMolGraphFeaturizer())
    dataset.cache = False
    return dataset


def make_loader(dataset, batch_size, shuffle=False, seed=17):
    # Explicit drop_last=False prevents Chemprop's singleton-dropping convenience default.
    return DataLoader(dataset, batch_size=batch_size, shuffle=shuffle, num_workers=0,
                      collate_fn=data.collate_batch, drop_last=False,
                      generator=torch.Generator().manual_seed(seed))


def make_model(config, pilot=False):
    return models.MPNN(
        nn.BondMessagePassing(d_h=config["hidden_dim"], depth=config["depth"], dropout=config["dropout"]),
        nn.MeanAggregation(),
        nn.BinaryClassificationFFN(n_tasks=1, input_dim=config["hidden_dim"], hidden_dim=config["hidden_dim"],
                                   n_layers=config["ffn_hidden_layers"], dropout=config["dropout"]),
        batch_norm=False, metrics=[AveragePrecision()],
        warmup_epochs=config["pilot"]["warmup_epochs"] if pilot else config["warmup_epochs"],
        init_lr=config["init_lr"], max_lr=config["max_lr"], final_lr=config["final_lr"])


class Progress(pl.Callback):
    def __init__(self, destination, pilot, budget):
        self.destination, self.pilot, self.budget = destination, pilot, budget
        self.epochs = []

    def on_train_epoch_start(self, trainer, pl_module):
        self.started = time.perf_counter()

    def on_train_batch_end(self, trainer, pl_module, outputs, batch, batch_idx):
        if not torch.isfinite(outputs["loss"]).all():
            raise RuntimeError("Non-finite training loss")
        if batch_idx % 100 == 0 and runtime()["peak_rss_mib"] > self.budget:
            raise RuntimeError("Measured process RSS exceeded the local resource budget")

    def on_validation_end(self, trainer, pl_module):
        if trainer.sanity_checking:
            return
        record = {"epoch": trainer.current_epoch + 1, "epoch_seconds_including_validation": time.perf_counter() - self.started,
                  "peak_rss_mib": runtime()["peak_rss_mib"]}
        if not self.pilot:
            record["validation_ap"] = float(trainer.callback_metrics["val/ap"])
            record["validation_loss"] = float(trainer.callback_metrics["val_loss"])
            if not np.isfinite(record["validation_ap"]) or not np.isfinite(record["validation_loss"]):
                raise RuntimeError("Non-finite validation metric")
        self.epochs.append(record)
        save_json(self.destination / "epoch_progress.json", self.epochs)
        print(json.dumps(record), flush=True)


def main():
    parser = argparse.ArgumentParser()
    parser.add_argument("--run-id", required=True)
    parser.add_argument("--pilot", action="store_true")
    parser.add_argument("--seed", type=int, default=17)
    args = parser.parse_args()
    if not args.run_id or any(c not in "abcdefghijklmnopqrstuvwxyzABCDEFGHIJKLMNOPQRSTUVWXYZ0123456789-_" for c in args.run_id):
        raise ValueError("Invalid run id")
    config_path, base_path = ROOT / "configs/chemprop_v0.json", ROOT / "configs/v0.json"
    config, base = json.loads(config_path.read_text()), json.loads(base_path.read_text())
    if config["final_test_evaluation"] is not False or base["final_test_evaluation"] is not False:
        raise ValueError("Development-only runner")
    if args.seed not in config["training_seeds"]:
        raise ValueError("Seed must be declared before evaluating results")
    if importlib.metadata.version("chemprop") != config["chemprop_version"]:
        raise ValueError("Chemprop version differs from the declared experiment")
    if (config["accelerator"], config["num_workers"], config["cache_graphs"], config["batch_norm"],
        config["extra_descriptors"], config["class_balancing"], config["loss"]) != (
        "cpu", 0, False, False, False, False, "unweighted_binary_cross_entropy_with_logits"):
        raise ValueError("Unsupported configuration; implement and verify any protocol changes explicitly")
    torch.set_num_threads(config["cpu_threads"])
    torch.set_num_interop_threads(1)
    pl.seed_everything(args.seed, workers=True)
    start = time.perf_counter()
    destination = ROOT / "outputs" / args.run_id
    destination.mkdir(parents=True, exist_ok=False)
    processed = ROOT / "data/processed/wong_v0"
    manifest = json.loads((processed / "manifest.json").read_text())
    if manifest["config"] != base:
        raise ValueError("Dataset configuration changed")
    rows = {}
    for partition in ("train", "validation"):
        name = f"{partition}.parquet"
        if sha256(processed / name) != manifest["output_hashes"][name]:
            raise ValueError(f"Data content changed: {partition}")
        rows[partition] = pq.read_table(processed / name).to_pylist()
    full_sizes = {k: len(v) for k, v in rows.items()}
    # Largest actual training molecules exercise a deliberately difficult minibatch.
    largest = sorted(rows["train"], key=lambda r: (-r["heavy_atoms"], r["compound_key"]))[:config["batch_size"]]
    if args.pilot:
        rng = np.random.default_rng(config["pilot"]["sample_seed"])
        for partition in rows:
            selected = rng.choice(len(rows[partition]), min(config["pilot"][f"{partition}_n"], len(rows[partition])), replace=False)
            rows[partition] = [rows[partition][int(i)] for i in selected]
    train, validation = rows["train"], rows["validation"]
    result = {"run_id": args.run_id, "created_at_utc": datetime.now(timezone.utc).isoformat(),
              "pilot_only": args.pilot, "evaluation_partition": "validation", "final_test_evaluated": False,
              "config": config, "config_sha256": sha256(config_path), "seed": args.seed,
              "dataset_manifest_sha256": sha256(processed / "manifest.json"), "source_code_hashes": source_hashes(),
              "full_sizes": full_sizes, "training_compounds": len(train), "validation_compounds": len(validation),
              "training_hits": sum(r["activity"] for r in train), "validation_hits": sum(r["activity"] for r in validation),
              "runtime_at_start": machine_runtime()}
    save_json(destination / "run_started.json", result)
    if args.pilot:
        model = make_model(config, pilot=True)
        batch = next(iter(make_loader(make_dataset(largest), config["batch_size"])))
        stress_start = time.perf_counter()
        bmg, vd, xd, targets, *_ = batch
        logits = model.predictor.train_step(model.fingerprint(bmg, vd, xd))
        loss = torch.nn.functional.binary_cross_entropy_with_logits(logits, targets)
        loss.backward()
        if not torch.isfinite(loss):
            raise RuntimeError("Non-finite stress-batch loss")
        result["largest_training_batch_probe"] = {"compound_keys": [r["compound_key"] for r in largest],
            "heavy_atoms": [r["heavy_atoms"] for r in largest], "seconds": time.perf_counter() - stress_start,
            "peak_rss_mib": runtime()["peak_rss_mib"]}
        del model, batch, logits, loss, bmg
        pl.seed_everything(args.seed, workers=True)
    build_start = time.perf_counter()
    train_loader = make_loader(make_dataset(train), config["batch_size"], shuffle=True, seed=args.seed)
    val_loader = make_loader(make_dataset(validation), config["batch_size"])
    result["dataset_build_seconds"] = time.perf_counter() - build_start
    model = make_model(config, args.pilot)
    result["parameter_count"] = sum(p.numel() for p in model.parameters())
    progress = Progress(destination, args.pilot, config["process_memory_budget_mib"])
    callbacks = [progress]
    if not args.pilot:
        checkpoint = ModelCheckpoint(dirpath=destination / "checkpoints", filename="best-{epoch:02d}", monitor="val/ap",
                                     mode="max", save_top_k=1, save_last=False)
        callbacks += [checkpoint, EarlyStopping(monitor="val/ap", mode="max", patience=config["early_stopping_patience"], check_finite=True)]
    trainer = pl.Trainer(accelerator="cpu", devices=1, precision="32-true", deterministic=True,
                         max_epochs=config["pilot"]["epochs"] if args.pilot else config["max_epochs"],
                         min_epochs=1 if args.pilot else config["min_epochs"], callbacks=callbacks,
                         logger=False if args.pilot else CSVLogger(str(destination), name="training", version="0"),
                         enable_checkpointing=not args.pilot, enable_progress_bar=False, enable_model_summary=False,
                         num_sanity_val_steps=0, log_every_n_steps=50)
    fit_start = time.perf_counter()
    trainer.fit(model, train_loader, val_loader)
    result.update({"fit_seconds": time.perf_counter() - fit_start, "epochs": progress.epochs})
    if not args.pilot:
        # This file was produced locally above; no third-party checkpoint is unpickled.
        saved = torch.load(checkpoint.best_model_path, map_location="cpu", weights_only=False)
        model.load_state_dict(saved["state_dict"])
        result["selected_epoch"] = int(saved["epoch"]) + 1
        result["checkpoint_sha256"] = sha256(checkpoint.best_model_path)
        result["checkpoint_path"] = str(os.path.relpath(checkpoint.best_model_path, ROOT))
        scores = torch.cat(trainer.predict(model, val_loader)).detach().cpu().numpy().reshape(-1)
        if len(scores) != len(validation) or not np.isfinite(scores).all() or np.any((scores < 0) | (scores > 1)):
            raise ValueError("Invalid/incomplete validation predictions")
        labels, keys = [r["activity"] for r in validation], [r["compound_key"] for r in validation]
        result["metrics"] = metrics(labels, scores, keys, base["top_k"])
        if abs(result["metrics"]["average_precision"] - float(checkpoint.best_model_score)) > 1e-6:
            raise ValueError("Saved-model AP disagrees with epoch selection metric")
        write_predictions(destination / "chemprop_validation.csv", validation, labels, scores)
    else:
        seconds = np.median([r["epoch_seconds_including_validation"] for r in progress.epochs])
        scale = sum(full_sizes.values()) / (len(train) + len(validation))
        result["rough_full_20_epoch_minutes"] = float(seconds * scale * config["max_epochs"] / 60)
        result["estimate_caveat"] = "Linear extrapolation of mixed train/validation throughput; full dataset startup and thermal effects are unmeasured."
    result.update({"complete": True, "total_seconds": time.perf_counter() - start, "runtime": machine_runtime()})
    save_json(destination / "results.json", result)
    print(json.dumps({k: result[k] for k in ("run_id", "complete", "total_seconds", "runtime", "final_test_evaluated")}), flush=True)


if __name__ == "__main__":
    main()
