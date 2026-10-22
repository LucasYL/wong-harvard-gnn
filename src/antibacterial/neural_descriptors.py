"""Light Chemprop + 200 global descriptors; development data only."""
import os
for variable in ("OMP_NUM_THREADS", "OPENBLAS_NUM_THREADS", "MKL_NUM_THREADS", "VECLIB_MAXIMUM_THREADS"):
    os.environ.setdefault(variable, "2")

import argparse
import json
import time
from datetime import datetime, timezone

import numpy as np
import torch
from chemprop import data, featurizers, models, nn
from lightning import pytorch as pl
from lightning.pytorch.callbacks import EarlyStopping, ModelCheckpoint
from lightning.pytorch.loggers import CSVLogger
from sklearn.metrics import auc, precision_recall_curve

from .baseline import metrics, write_predictions
from .common import ROOT, save_json, sha256, source_hashes
from .descriptors import CONFIG, CACHE, read_rows, load_features
from .neural import Progress, make_loader, machine_runtime
from .neural_metrics import AveragePrecision


def dataset(rows, features):
    if features.shape != (len(rows), 200) or not np.isfinite(features).all():
        raise ValueError("Invalid row-aligned descriptor matrix")
    points = [data.MoleculeDatapoint.from_smi(
        r["parent_smiles"], np.asarray([r["activity"]], dtype=np.float32), x_d=x.copy())
        for r, x in zip(rows, features)]
    result = data.MoleculeDataset(points, featurizers.SimpleMoleculeMolGraphFeaturizer())
    result.cache = False
    return result


def model(config, pilot=False):
    return models.MPNN(
        nn.BondMessagePassing(d_h=config["hidden_dim"], depth=config["depth"], dropout=config["dropout"]),
        nn.MeanAggregation(),
        nn.BinaryClassificationFFN(n_tasks=1, input_dim=config["hidden_dim"] + 200,
            hidden_dim=config["hidden_dim"], n_layers=config["ffn_hidden_layers"], dropout=config["dropout"]),
        batch_norm=False, metrics=[AveragePrecision()],
        warmup_epochs=config["pilot"]["warmup_epochs"] if pilot else config["warmup_epochs"],
        init_lr=config["init_lr"], max_lr=config["max_lr"], final_lr=config["final_lr"])


def all_metrics(labels, scores, keys, top_k):
    result = metrics(labels, scores, keys, top_k)
    precision, recall, _ = precision_recall_curve(labels, scores)
    result["trapezoidal_pr_auc"] = float(auc(recall, precision))
    return result


def mean_scores(members):
    values = np.asarray(members, dtype=np.float64)
    if values.ndim != 2 or min(values.shape) == 0 or not np.isfinite(values).all():
        raise ValueError("Need finite, row-aligned member predictions")
    if np.any((values < 0) | (values > 1)):
        raise ValueError("Member score outside [0, 1]")
    return values.mean(axis=0)


class RecoverableProgress(Progress):
    def __init__(self, destination, pilot, budget, loader):
        super().__init__(destination, pilot, budget)
        self.loader = loader

    def on_save_checkpoint(self, trainer, pl_module, checkpoint):
        checkpoint["descriptor_loader_rng"] = self.loader.generator.get_state()
        checkpoint["recorded_epochs"] = self.epochs

    def on_load_checkpoint(self, trainer, pl_module, checkpoint):
        if "descriptor_loader_rng" not in checkpoint:
            raise ValueError("Resume checkpoint lacks loader state")
        self.loader.generator.set_state(checkpoint["descriptor_loader_rng"])
        self.epochs = checkpoint["recorded_epochs"]


def main():
    parser = argparse.ArgumentParser()
    parser.add_argument("--run-id", required=True)
    parser.add_argument("--seed", type=int, required=True)
    parser.add_argument("--pilot", action="store_true")
    parser.add_argument("--resume", action="store_true")
    args = parser.parse_args()
    if not args.run_id or any(c not in "abcdefghijklmnopqrstuvwxyzABCDEFGHIJKLMNOPQRSTUVWXYZ0123456789-_" for c in args.run_id):
        raise ValueError("Invalid run ID")
    config = json.loads(CONFIG.read_text())
    if args.seed not in config["training_seeds"] or config["final_test_evaluation"]:
        raise ValueError("Undeclared seed or forbidden final-test evaluation")
    if (config["aggregation"], config["descriptor_dim"], config["accelerator"], config["batch_norm"],
        config["class_balancing"], config["extra_descriptors"]) != ("mean", 200, "cpu", False, False, True):
        raise ValueError("Unsupported model configuration")
    torch.set_num_threads(config["cpu_threads"])
    torch.set_num_interop_threads(1)
    pl.seed_everything(args.seed, workers=True)
    start = time.perf_counter()
    destination = ROOT / "outputs" / args.run_id
    destination.mkdir(parents=True, exist_ok=args.resume)
    if (destination / "results.json").exists():
        raise ValueError("Completed run cannot be overwritten")
    rows = {p: read_rows(p, labels=True) for p in ("train", "validation")}
    features = {p: load_features(p, rows[p]) for p in rows}
    full_sizes = {p: len(r) for p, r in rows.items()}
    base = json.loads((ROOT / "configs/v0.json").read_text())
    identity = {"config_sha256": sha256(CONFIG), "descriptor_manifest_sha256": sha256(CACHE / "manifest.json"),
                "dataset_manifest_sha256": sha256(ROOT / "data/processed/wong_v0/manifest.json"),
                "source_code_hashes": source_hashes()}
    result = {"run_id": args.run_id, "seed": args.seed, "pilot_only": args.pilot,
              "created_at_utc": datetime.now(timezone.utc).isoformat(), "config": config, **identity,
              "evaluation_partition": "validation", "final_test_evaluated": False,
              "runtime_at_start": machine_runtime(), "full_sizes": full_sizes, "resumed": args.resume}
    if result["runtime_at_start"]["packages"]["chemprop"] != config["chemprop_version"]:
        raise ValueError("Chemprop version changed")
    if not args.pilot:
        protocol = json.loads((ROOT / "configs/descriptor_experiment_frozen_v1.json").read_text())
        if protocol["identity"] != identity:
            raise ValueError("Experiment differs from the frozen protocol")
    if args.resume:
        previous = json.loads((destination / "run_started.json").read_text())
        if any(previous[k] != v for k, v in identity.items()) or previous["seed"] != args.seed:
            raise ValueError("Cannot resume with different code, inputs or seed")
        save_json(destination / f"resume_{time.time_ns()}.json", result)
    else:
        save_json(destination / "run_started.json", result)
    if args.pilot:
        indices = sorted(range(len(rows["train"])), key=lambda i: -rows["train"][i]["heavy_atoms"])[:config["batch_size"]]
        probe = model(config, True)
        batch = next(iter(make_loader(dataset([rows["train"][i] for i in indices], features["train"][indices]), config["batch_size"])))
        bmg, vd, xd, targets, *_ = batch
        logits = probe.predictor.train_step(probe.fingerprint(bmg, vd, xd))
        loss = torch.nn.functional.binary_cross_entropy_with_logits(logits, targets)
        loss.backward()
        if not torch.isfinite(loss):
            raise ValueError("Non-finite largest-molecule batch")
        result["stress_max_heavy_atoms"] = max(rows["train"][i]["heavy_atoms"] for i in indices)
        del probe, batch, logits, loss, bmg, vd, xd, targets
        rng = np.random.default_rng(config["pilot"]["sample_seed"])
        for p in rows:
            indices = rng.choice(len(rows[p]), config["pilot"][f"{p}_n"], replace=False)
            rows[p] = [rows[p][i] for i in indices]
            features[p] = features[p][indices]
        pl.seed_everything(args.seed, workers=True)
    train_loader = make_loader(dataset(rows["train"], features["train"]), config["batch_size"], shuffle=True, seed=args.seed)
    val_loader = make_loader(dataset(rows["validation"], features["validation"]), config["batch_size"])
    network = model(config, args.pilot)
    result["parameter_count"] = sum(p.numel() for p in network.parameters())
    progress = RecoverableProgress(destination, args.pilot, config["process_memory_budget_mib"], train_loader)
    callbacks = [progress]
    if not args.pilot:
        checkpoint = ModelCheckpoint(dirpath=destination / "checkpoints", filename="best-{epoch:02d}",
            monitor="val/ap", mode="max", save_top_k=1, save_last=True)
        callbacks += [checkpoint, EarlyStopping(monitor="val/ap", mode="max",
            patience=config["early_stopping_patience"], check_finite=True)]
    trainer = pl.Trainer(accelerator="cpu", devices=1, precision="32-true", deterministic=True,
        max_epochs=config["pilot"]["epochs"] if args.pilot else config["max_epochs"],
        min_epochs=1 if args.pilot else config["min_epochs"], callbacks=callbacks,
        logger=False if args.pilot else CSVLogger(str(destination), name="training", version="0"),
        enable_checkpointing=not args.pilot, enable_progress_bar=False, enable_model_summary=False,
        num_sanity_val_steps=0, log_every_n_steps=50)
    last = destination / "checkpoints/last.ckpt"
    if args.resume and not last.exists():
        raise ValueError("No resumable epoch checkpoint; preserve this attempt and use a new run ID")
    fit_start = time.perf_counter()
    trainer.fit(network, train_loader, val_loader, ckpt_path=str(last) if args.resume else None)
    result.update(fit_seconds=time.perf_counter() - fit_start, epochs=progress.epochs)
    if not args.pilot:
        saved = torch.load(checkpoint.best_model_path, map_location="cpu", weights_only=False)
        network.load_state_dict(saved["state_dict"])
        result.update(selected_epoch=int(saved["epoch"]) + 1, checkpoint_sha256=sha256(checkpoint.best_model_path),
                      checkpoint_path=os.path.relpath(checkpoint.best_model_path, ROOT))
        scores = torch.cat(trainer.predict(network, val_loader)).detach().cpu().numpy().reshape(-1)
        validation = rows["validation"]
        labels, keys = [r["activity"] for r in validation], [r["compound_key"] for r in validation]
        if len(scores) != len(validation) or not np.isfinite(scores).all() or np.any((scores < 0) | (scores > 1)):
            raise ValueError("Invalid prediction coverage")
        result["metrics"] = all_metrics(labels, scores, keys, base["top_k"])
        if abs(result["metrics"]["average_precision"] - float(checkpoint.best_model_score)) > 1e-6:
            raise ValueError("Reloaded checkpoint does not reproduce selection AP")
        write_predictions(destination / "chemprop_validation.csv", validation, labels, scores)
    else:
        scale = sum(full_sizes.values()) / sum(map(len, rows.values()))
        seconds = float(np.median([e["epoch_seconds_including_validation"] for e in progress.epochs]))
        result["rough_full_model_minutes"] = seconds * scale * config["max_epochs"] / 60
        result["estimate_caveat"] = "Linear pilot extrapolation; full-data startup, large-molecule distribution and thermal effects may differ."
    if identity != {"config_sha256": sha256(CONFIG), "descriptor_manifest_sha256": sha256(CACHE / "manifest.json"),
                    "dataset_manifest_sha256": sha256(ROOT / "data/processed/wong_v0/manifest.json"),
                    "source_code_hashes": source_hashes()}:
        raise ValueError("Code or data identity changed during training")
    # Verify the feature files and source rows again before reporting success.
    for p in ("train", "validation"):
        load_features(p, read_rows(p))
    result.update(complete=True, total_seconds=time.perf_counter() - start, runtime=machine_runtime())
    save_json(destination / "results.json", result)
    print(json.dumps({k: result[k] for k in ("run_id", "complete", "total_seconds", "runtime", "final_test_evaluated")}), flush=True)


if __name__ == "__main__":
    main()
