import argparse
import csv
import json
import time
from collections import Counter

import pyarrow as pa
import pyarrow.parquet as pq
from rdkit import RDLogger

from .chemistry import consolidate, standardize
from .common import ROOT, runtime, save_json, sha256, source_hashes
from .splitting import scaffold_split


def main():
    parser = argparse.ArgumentParser()
    parser.add_argument("--limit", type=int, help="Resource pilot only; does not create splits or training data")
    args = parser.parse_args()
    started = time.perf_counter()
    config = json.loads((ROOT / "configs/v0.json").read_text())
    source = ROOT / "data/interim/wong_measurements.jsonl"
    source_audit = json.loads((ROOT / "reports/source_audit.json").read_text())
    assert sha256(source) == source_audit["records_sha256"]
    records = [json.loads(line) for line in source.read_text().splitlines()]
    if args.limit:
        # Evenly-spaced rows avoid testing only the hit-enriched beginning of the workbook.
        step = max(1, len(records) // args.limit)
        records = records[::step][:args.limit]
    RDLogger.DisableLog("rdApp.warning")
    for i, row in enumerate(records, 1):
        row.update(standardize(row["raw_smiles"]))
        if i % 5000 == 0:
            print(f"Processed {i:,} structures in {time.perf_counter() - started:.1f}s", flush=True)
    compounds = consolidate(records)
    stats = {"source_records": len(records), "eligible_compounds": len(compounds), "eligible_hits": sum(r["activity"] for r in compounds),
             "eligible_records": sum(r["eligible"] for r in records), "excluded_records": sum(not r["eligible"] for r in records),
             "structure_flags": dict(Counter(f for r in records for f in r["structure_flags"])),
             "structure_changed_records": sum(r.get("structure_changed", False) for r in records),
             "conflicting_parent_groups": len({r["parent_smiles"] for r in records if r.get("conflicting_parent_labels")}),
             "conflicting_parent_records": sum(r.get("conflicting_parent_labels", False) for r in records),
             "duplicate_eligible_records_collapsed": sum(r["eligible"] for r in records) - len(compounds),
             "ambiguous_inchikey_groups": len({r["parent_inchikey"] for r in records if "same_inchikey_different_parent_smiles" in r["structure_flags"]}),
             "unique_scaffolds": len({r["scaffold"] for r in compounds}),
             "acyclic_compounds": sum(r["scaffold"] == "ACYCLIC" for r in compounds),
             "max_heavy_atoms": max(r["heavy_atoms"] for r in compounds),
             "natural_product_annotation": "unknown; not inferred from structures or compound names"}
    if args.limit:
        stats.update({"pilot_only": True, "seconds": time.perf_counter() - started, "runtime": runtime()})
        save_json(ROOT / "reports/structure_pilot.json", stats)
        print(json.dumps(stats, indent=2))
        return
    save_json(ROOT / "reports/structure_audit_before_split.json", {**stats, "seconds": time.perf_counter() - started, "runtime": runtime()})
    stats["splits"] = scaffold_split(compounds, config["split_seed"], config["split_fractions"])
    if any(s["hits"] < 20 or s["compounds"] - s["hits"] < 20 for s in stats["splits"].values()):
        raise ValueError("A split has fewer than 20 hits/non-hits. Inspect before evaluation; no automatic reseeding.")
    dest = ROOT / "data/processed/wong_v0"
    dest.mkdir(parents=True, exist_ok=True)
    # Check membership before writing any frozen output.
    assignments = [{"compound_key": r["compound_key"], "split": r["split"], "scaffold": r["scaffold"]} for r in compounds]
    split_text = "".join(json.dumps(r, sort_keys=True) + "\n" for r in assignments)
    split_path = dest / "split_assignments.jsonl"
    if split_path.exists() and split_path.read_text() != split_text:
        raise RuntimeError("Frozen split differs. Create a new version; do not overwrite.")
    split_path.write_text(split_text)
    pq.write_table(pa.Table.from_pylist(records), dest / "measurements.parquet", compression="zstd")
    for name in ("train", "validation", "test"):
        pq.write_table(pa.Table.from_pylist([r for r in compounds if r["split"] == name]), dest / f"{name}.parquet", compression="zstd")
    with (dest / "review_records.jsonl").open("w") as handle:
        for row in records:
            if not row["eligible"] or row["measurement_flags"]:
                handle.write(json.dumps(row) + "\n")
    # A compact, auditable selection for manual source checks, including every exclusion category.
    selected = {}
    for row in records:
        reasons = row["structure_flags"] + row["measurement_flags"] + (["conflicting_parent_labels"] if row.get("conflicting_parent_labels") else [])
        if not reasons:
            reasons = [f"ordinary_label_{row['activity']}"]
        for reason in reasons:
            selected.setdefault(reason, [])
            if len(selected[reason]) < 2:
                selected[reason].append(row)
    save_json(ROOT / "reports/provenance_examples.json", selected)
    stats.update({"config": config, "measurement_source_sha256": sha256(source), "raw_manifest_sha256": sha256(ROOT / "data/raw/wong2024/manifest.json"),
                  "source_code_hashes": source_hashes(), "seconds": time.perf_counter() - started, "runtime": runtime(),
                  "output_hashes": {p.name: sha256(p) for p in sorted(dest.iterdir()) if p.is_file() and p.name != "manifest.json"}})
    save_json(ROOT / "reports/dataset_audit.json", stats)
    save_json(dest / "manifest.json", stats)
    print(json.dumps({k: v for k, v in stats.items() if k not in ("source_code_hashes", "output_hashes")}, indent=2))


if __name__ == "__main__":
    main()
