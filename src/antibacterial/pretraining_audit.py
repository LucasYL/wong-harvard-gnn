"""Resume a structure-only overlap audit without opening held-out activity labels."""
import argparse
from collections import Counter
from concurrent.futures import ProcessPoolExecutor
import itertools
import json
import os
import resource
import sqlite3
import sys
import time

from rdkit import RDLogger

from .chemistry import standardize
from .common import ROOT, runtime, save_json, sha256


def identify(item):
    line, smiles = item
    RDLogger.DisableLog("rdApp.warning")
    RDLogger.DisableLog("rdApp.error")
    result = standardize(smiles)
    peak = resource.getrusage(resource.RUSAGE_SELF).ru_maxrss / (1024**2 if sys.platform == "darwin" else 1024)
    return line, smiles.strip(), result, os.getpid(), peak


def main():
    parser = argparse.ArgumentParser()
    parser.add_argument("--pilot", action="store_true")
    parser.add_argument("--run-id", required=True)
    parser.add_argument("--workers", type=int, choices=(1, 2), default=2)
    args = parser.parse_args()
    if not args.run_id or any(c not in "abcdefghijklmnopqrstuvwxyzABCDEFGHIJKLMNOPQRSTUVWXYZ0123456789-_" for c in args.run_id):
        raise ValueError("Invalid audit run id")
    source = ROOT / "data/raw/chemeleon/cleaned_pubchem_1MM.smiles"
    assignments_path = ROOT / "data/processed/wong_v0/split_assignments.jsonl"
    raw_manifest = json.loads((source.parent / "manifest.json").read_text())
    expected = next(f["sha256"] for f in raw_manifest["files"] if f["file"] == source.name)
    if sha256(source) != expected:
        raise ValueError("Pretraining source content changed")
    manifest = json.loads((assignments_path.parent / "manifest.json").read_text())
    if sha256(assignments_path) != manifest["output_hashes"][assignments_path.name]:
        raise ValueError("Frozen assignments changed")
    assignments = [json.loads(line) for line in assignments_path.read_text().splitlines()]
    by_key = {r["compound_key"]: r["split"] for r in assignments}
    connectivity = {key[:14] for key in by_key}
    scaffolds = {r["scaffold"] for r in assignments}
    identity = {"source_sha256": expected, "assignments_sha256": sha256(assignments_path),
                "standardizer_sha256": sha256(ROOT / "src/antibacterial/chemistry.py"),
                "audit_code_sha256": sha256(__file__), "rdkit": runtime()["packages"]["rdkit"], "pilot": args.pilot}
    destination = ROOT / "data/processed" / args.run_id
    destination.mkdir(parents=True, exist_ok=True)
    db = sqlite3.connect(destination / "audit.sqlite")
    db.execute("CREATE TABLE IF NOT EXISTS state (id INTEGER PRIMARY KEY, value TEXT NOT NULL)")
    db.execute("CREATE TABLE IF NOT EXISTS evidence (line INTEGER PRIMARY KEY, smiles TEXT, full_key TEXT, scaffold TEXT, eligible INTEGER, flags TEXT)")
    existing = db.execute("SELECT value FROM state WHERE id=1").fetchone()
    state = json.loads(existing[0]) if existing else {"identity": identity, "processed": 0, "eligible": 0, "flag_counts": {}, "worker_peak_rss_mib": {}, "processing_seconds": 0.0, "complete": False}
    completed_path = destination / "results.json"
    published_path = ROOT / "reports" / (args.run_id + ".json")
    if state["complete"] and completed_path.exists():
        # Read a finished historical result without overwriting its measured runtime.
        for key in ("source_sha256", "assignments_sha256", "standardizer_sha256", "rdkit", "pilot"):
            if state["identity"][key] != identity[key]:
                raise ValueError("Completed audit inputs or chemistry protocol changed")
        finished = json.loads(completed_path.read_text())
        if not finished["complete"] or finished["identity"] != state["identity"]:
            raise ValueError("Completed result and audit state disagree")
        if published_path.exists() and published_path.read_bytes() != completed_path.read_bytes():
            raise ValueError("Published audit report differs from its original result")
        if not published_path.exists():
            published_path.write_bytes(completed_path.read_bytes())
        db.close()
        print(json.dumps({"complete": True, "reused_finished_result": True, "processed": state["processed"],
                          "execution_code_sha256": state["identity"]["audit_code_sha256"]}), flush=True)
        return
    if state["identity"] != identity:
        raise ValueError("Audit inputs/code changed; create a new audit version")
    def save_state():
        db.execute("INSERT OR REPLACE INTO state VALUES (1, ?)", (json.dumps(state),))
        db.commit()
        save_json(destination / "progress.json", state)
    start = time.perf_counter()
    accumulated_seconds = state["processing_seconds"]
    flag_counts = Counter(state["flag_counts"])
    if not state["complete"]:
        with source.open() as handle, ProcessPoolExecutor(max_workers=args.workers) as pool:
            # Pilot is evenly spaced across the file to avoid its ordering bias.
            if args.pilot:
                total = sum(1 for _ in handle)
                handle.seek(0)
                step = max(1, total // 1000)
                items = itertools.islice(((i, text) for i, text in enumerate(handle, 1) if (i - 1) % step == 0), 1000)
                items = itertools.islice(items, state["processed"], None)
            else:
                items = itertools.islice(enumerate(handle, 1), state["processed"], None)
            while batch := list(itertools.islice(items, 2000)):
                for line, smiles, result, pid, peak in pool.map(identify, batch, chunksize=50):
                    state["worker_peak_rss_mib"][str(pid)] = max(peak, state["worker_peak_rss_mib"].get(str(pid), 0))
                    state["processed"] += 1
                    state["eligible"] += int(result["structure_eligible"])
                    flag_counts.update(result["structure_flags"])
                    key, scaffold = result.get("parent_inchikey", ""), result.get("scaffold", "")
                    if key in by_key or key[:14] in connectivity or scaffold in scaffolds or result["structure_flags"]:
                        db.execute("INSERT OR REPLACE INTO evidence VALUES (?, ?, ?, ?, ?, ?)",
                                   (line, smiles, key, scaffold, int(result["structure_eligible"]), json.dumps(result["structure_flags"])))
                state.update({"flag_counts": dict(flag_counts), "processing_seconds": accumulated_seconds + time.perf_counter() - start})
                save_state()
                print(json.dumps({"processed": state["processed"], "seconds": state["processing_seconds"]}), flush=True)
        state["complete"] = True
        save_state()
    exact, connected, seen_scaffolds = set(), set(), set()
    ambiguous_exact = set()
    for key, scaffold, eligible in db.execute("SELECT full_key, scaffold, eligible FROM evidence"):
        if eligible:
            if key in by_key:
                exact.add(key)
            connected.add(key[:14])
            seen_scaffolds.add(scaffold)
        elif key in by_key:
            ambiguous_exact.add(key)
    overlaps = {}
    for partition in ("train", "validation", "test"):
        keys = {r["compound_key"] for r in assignments if r["split"] == partition}
        overlap = keys & exact
        overlaps[partition] = {"compounds": len(keys), "exact_parent_identity_exposed": len(overlap),
                              "connectivity_identity_exposed": sum(k[:14] in connected for k in keys),
                              "scaffold_exposed_compounds": sum(r["scaffold"] in seen_scaffolds for r in assignments if r["split"] == partition),
                              "ambiguous_parent_matches": len(keys & ambiguous_exact), "exact_compound_keys": sorted(overlap)}
    report = {**state, "workers": args.workers, "main_process_runtime": runtime(), "activity_labels_read": False,
              "final_test_predictions_or_performance": False, "overlaps": overlaps,
              "interpretation": "Structural exposure only. No target assay labels are used. Ineligible source structures remain uncertain; non-matching identities are not proof that the model has seen no related chemical structure."}
    report["sum_of_main_and_largest_worker_peaks_mib"] = runtime()["peak_rss_mib"] + sum(sorted(state["worker_peak_rss_mib"].values(), reverse=True)[:args.workers])
    report["memory_caveat"] = "Sum of individual process peaks is a conservative estimate, not a simultaneous whole-machine RSS measurement."
    if args.pilot:
        report["estimated_full_minutes"] = state["processing_seconds"] / state["processed"] * 944296 / 60
        report["interpretation"] = "Resource pilot only; overlap counts concern this 1000-row sample and are not a completed corpus audit."
    save_json(destination / "results.json", report)
    save_json(ROOT / "reports" / (args.run_id + ".json"), report)
    db.close()
    print(json.dumps({"complete": True, "processed": state["processed"], "seconds": state["processing_seconds"],
                      "estimated_full_minutes": report.get("estimated_full_minutes"),
                      "overlaps": {p: {k: v for k, v in stats.items() if k != "exact_compound_keys"} for p, stats in overlaps.items()}}), flush=True)


if __name__ == "__main__":
    main()
