"""Read the untouched public CO-ADD CSV archive; inventory context, never make labels."""
import csv
import hashlib
import io
import json
import math
import resource
import sys
import time
import zipfile
from collections import Counter, defaultdict
from pathlib import Path

ROOT = Path(__file__).resolve().parents[1]
ARCHIVE = ROOT / "data/raw/assay_discovery/CO-ADD_InhibitionData_r03_01-02-2020_CSV.zip"
EXPECTED_SHA256 = "bd57000d373e1176940bfe9462bdaa21982612de776945dec167fa9ddcf864d2"


def main():
    start = time.perf_counter()
    digest = hashlib.sha256(ARCHIVE.read_bytes()).hexdigest()
    if digest != EXPECTED_SHA256:
        raise ValueError("Archive differs from the downloaded snapshot")
    organisms, missing = Counter(), Counter()
    groups = defaultdict(lambda: {"rows": 0, "ids": set(), "replicate_counts": Counter(), "nonfinite_inhibition": 0,
                                 "out_of_0_100_range": 0, "empty_smiles": 0})
    all_ids, sa_ids = set(), set()
    rows = 0
    with zipfile.ZipFile(ARCHIVE) as archive:
        members = archive.infolist()
        if len(members) != 1 or not members[0].filename.endswith(".csv") or members[0].file_size > 300_000_000:
            raise ValueError("Unexpected archive layout or size")
        with archive.open(members[0]) as binary:
            reader = csv.DictReader(io.TextIOWrapper(binary, encoding="utf-8-sig"))
            columns = reader.fieldnames
            for row in reader:
                if None in row or any(v is None for v in row.values()):
                    raise ValueError("Malformed CSV row")
                rows += 1
                organisms[row["ORGANISM"]] += 1
                all_ids.add(row["COADD_ID"])
                for name in ("COADD_ID", "ASSAY_ID", "ORGANISM", "STRAIN", "INHIB_AVE", "CONC", "SMILES"):
                    if not row[name].strip():
                        missing[name] += 1
                if row["ORGANISM"] != "Staphylococcus aureus":
                    continue
                sa_ids.add(row["COADD_ID"])
                group = groups[(row["ASSAY_ID"], row["STRAIN"], row["CONC"])]
                group["rows"] += 1
                group["ids"].add(row["COADD_ID"])
                group["replicate_counts"][row["NASSAYS"]] += 1
                group["empty_smiles"] += not row["SMILES"].strip()
                try:
                    value = float(row["INHIB_AVE"])
                except ValueError:
                    value = math.nan
                if not math.isfinite(value):
                    group["nonfinite_inhibition"] += 1
                elif value < 0 or value > 100:
                    group["out_of_0_100_range"] += 1
    records = [{"assay_id": key[0], "strain": key[1], "concentration": key[2],
                **{name: value for name, value in group.items() if name != "ids"},
                "distinct_source_ids": len(group["ids"])} for key, group in sorted(groups.items())]
    report = {"source_url": "https://db.co-add.org/javax.faces.resource/CO-ADD_InhibitionData_r03_01-02-2020_CSV.zip.xhtml?ln=files",
              "archive_sha256": digest, "archive_bytes": ARCHIVE.stat().st_size,
              "member": members[0].filename, "uncompressed_bytes": members[0].file_size,
              "rows": rows, "distinct_source_ids": len(all_ids), "organism_rows": organisms,
              "columns": columns, "missing_required_fields": missing, "saureus_distinct_source_ids": len(sa_ids),
              "saureus_context_groups": records, "new_training_labels_created": False,
              "structure_standardization_and_wong_overlap_completed": False,
              "limitations": ["Source IDs are not standardized compound identities.",
                              "Concentration units and strains differ from Wong RN4220 at 50 uM; do not pool labels.",
                              "Out-of-range inhibition is retained; negative growth inhibition is not missing data.",
                              "Incubation duration, medium, active-label rules and licensing still require source verification."],
              "seconds": time.perf_counter() - start,
              "peak_rss_mib": resource.getrusage(resource.RUSAGE_SELF).ru_maxrss / (1024**2 if sys.platform == "darwin" else 1024),
              "script_sha256": hashlib.sha256(Path(__file__).read_bytes()).hexdigest()}
    (ROOT / "reports/coadd_inventory.json").write_text(json.dumps(report, indent=2, ensure_ascii=False) + "\n")
    print(json.dumps(report, indent=2, ensure_ascii=False))


if __name__ == "__main__":
    main()
