"""Label-blind scaffold allocation. Test membership is never chosen by model scores."""
import hashlib
from collections import Counter, defaultdict


def scaffold_split(compounds, seed, fractions):
    if abs(sum(fractions.values()) - 1) > 1e-8:
        raise ValueError("Split fractions must add to one.")
    groups = defaultdict(list)
    for compound in compounds:
        groups[compound["scaffold"]].append(compound)
    # Hash ordering is stable across input ordering and Python releases.
    keys = sorted(groups, key=lambda key: hashlib.sha256(f"{seed}:{key}".encode()).hexdigest())
    n = len(compounds)
    test_end = round(n * fractions["test"])
    validation_end = test_end + round(n * fractions["validation"])
    assigned = 0
    for key in keys:
        split = "test" if assigned < test_end else "validation" if assigned < validation_end else "train"
        for row in groups[key]:
            row["split"] = split
        assigned += len(groups[key])
    assert_no_leakage(compounds)
    return {name: {"compounds": sum(r["split"] == name for r in compounds),
                   "hits": sum(r["activity"] for r in compounds if r["split"] == name),
                   "scaffolds": len({r["scaffold"] for r in compounds if r["split"] == name})}
            for name in ("train", "validation", "test")}


def assert_no_leakage(compounds):
    for field in ("compound_key", "parent_smiles", "connectivity_smiles", "scaffold"):
        membership = defaultdict(set)
        for row in compounds:
            membership[row[field]].add(row["split"])
        if any(len(splits) > 1 for splits in membership.values()):
            raise ValueError(f"Cross-split leakage in {field}")
    if set(r["split"] for r in compounds) != {"train", "validation", "test"}:
        raise ValueError("Empty split; inspect group sizes before defining a new split policy.")
