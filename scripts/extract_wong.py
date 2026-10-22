"""Read the unmodified supplementary workbook with bundled Python/openpyxl."""
import csv
import hashlib
import json
import math
import resource
import sys
import time
from collections import Counter, defaultdict
from pathlib import Path

import openpyxl

ROOT = Path(__file__).resolve().parents[1]
RAW = ROOT / "data/raw/wong2024"
SHEET = "S. aureus growth inhibition"
CUTOFF = 0.2  # Author's fixed screen definition, not a fitted threshold.


def main():
    started = time.perf_counter()
    manifest = json.loads((RAW / "manifest.json").read_text())
    for item in manifest["files"]:
        assert hashlib.sha256((RAW / item["name"]).read_bytes()).hexdigest() == item["sha256"]
    with (RAW / "author_train.csv").open(encoding="utf-8-sig", newline="") as handle:
        author = list(csv.DictReader(handle))
    mapping = defaultdict(list)
    for row_number, row in enumerate(author, 2):
        mapping[row["SMILES"].strip()].append((row_number, int(row["ACTIVITY"])))
    book = openpyxl.load_workbook(RAW / "supplementary_data_1.xlsx", read_only=True, data_only=True)
    sheet = book[SHEET]
    rows = sheet.iter_rows(values_only=True)
    assert tuple(next(rows)) == ("Compound_ID", "SMILES", "R1_50uM", "R2_50uM", "Mean_50uM")
    records = []
    for row_number, row in enumerate(rows, 2):
        compound_id, smiles, r1, r2, mean = row
        if not isinstance(smiles, str) or not all(isinstance(x, (int, float)) and math.isfinite(x) for x in (r1, r2, mean)):
            raise ValueError(f"Missing or nonnumeric source data at {SHEET}!A{row_number}:E{row_number}")
        matches = mapping.get(smiles.strip(), [])
        label = int(mean < CUTOFF)
        flags = []
        if not matches or any(x[1] != label for x in matches):
            flags.append("author_label_reconciliation_issue")
        if abs(mean - (r1 + r2) / 2) > 1e-7:
            flags.append("reported_mean_differs_from_replicates")
        if (r1 < CUTOFF) != (r2 < CUTOFF):
            flags.append("replicates_cross_hit_cutoff")
        if abs(mean - CUTOFF) <= 0.02:
            flags.append("near_cutoff_descriptive_only")
        records.append({"record_id": f"wong2024:{row_number}", "source_compound_id": str(compound_id),
                        "source_file": "supplementary_data_1.xlsx", "source_sheet": SHEET, "source_row": row_number,
                        "assay_id": "wong2024_saureus_rn4220_50uM", "doi": manifest["doi"],
                        "raw_smiles": smiles, "replicate_1_relative_growth": r1, "replicate_2_relative_growth": r2,
                        "mean_relative_growth": mean, "activity": label, "test_concentration": 50.0,
                        "concentration_unit": "uM", "endpoint": "relative_growth", "organism": "Staphylococcus aureus",
                        "strain": "RN4220", "relation": "=", "author_csv_rows": [x[0] for x in matches],
                        "author_csv_labels": sorted(set(x[1] for x in matches)), "measurement_flags": flags,
                        "np_status": "unknown", "source_organism": None, "actinomycete_status": "unknown"})
    workbook_pairs = Counter((r["raw_smiles"].strip(), r["activity"]) for r in records)
    author_pairs = Counter((r["SMILES"].strip(), int(r["ACTIVITY"])) for r in author)
    stats = {"source_rows": len(records), "author_csv_rows": len(author), "source_hits": sum(r["activity"] for r in records),
             "author_csv_hits": sum(int(r["ACTIVITY"]) for r in author), "multiset_matches_author_csv": workbook_pairs == author_pairs,
             "measurement_flags": dict(Counter(f for r in records for f in r["measurement_flags"])),
             "workbook_sheet": SHEET, "cutoff": CUTOFF, "comparison": "mean_relative_growth < cutoff",
             "near_cutoff_band": "0.18 through 0.22, descriptive only; does not change labels or eligibility",
             "source_id_duplicates": len(records) - len(set(r["source_compound_id"] for r in records)),
             "openpyxl_version": openpyxl.__version__, "python_version": sys.version,
             "seconds": time.perf_counter() - started,
             "peak_rss_mib": resource.getrusage(resource.RUSAGE_SELF).ru_maxrss / (1024**2 if sys.platform == "darwin" else 1024)}
    if not stats["multiset_matches_author_csv"]:
        raise ValueError("Workbook and author CSV disagree; investigate before training.")
    dest = ROOT / "data/interim"
    dest.mkdir(parents=True, exist_ok=True)
    path = dest / "wong_measurements.jsonl"
    path.write_text("".join(json.dumps(r, ensure_ascii=False) + "\n" for r in records))
    stats["records_sha256"] = hashlib.sha256(path.read_bytes()).hexdigest()
    report = ROOT / "reports"
    report.mkdir(exist_ok=True)
    (report / "source_audit.json").write_text(json.dumps(stats, indent=2, ensure_ascii=False) + "\n")
    book.close()
    print(json.dumps(stats, indent=2, ensure_ascii=False))


if __name__ == "__main__":
    main()
