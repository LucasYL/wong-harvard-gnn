"""Conservative structure identities; ambiguous records remain auditable."""
from collections import defaultdict

from rdkit import Chem
from rdkit.Chem import Descriptors
from rdkit.Chem.MolStandardize import rdMolStandardize
from rdkit.Chem.Scaffolds import MurckoScaffold

UNCHARGER = rdMolStandardize.Uncharger()


def standardize(smiles):
    result = {"structure_flags": []}
    mol = Chem.MolFromSmiles(smiles.strip())
    if mol is None:
        return {**result, "structure_flags": ["invalid_smiles"], "structure_eligible": False}
    try:
        result["canonical_raw_smiles"] = Chem.MolToSmiles(mol, isomericSmiles=True)
        clean = rdMolStandardize.Cleanup(mol)
        fragments = Chem.GetMolFrags(clean, asMols=True, sanitizeFrags=True)
        organic = [f for f in fragments if any(a.GetAtomicNum() == 6 for a in f.GetAtoms())]
        # A conservative rule avoids guessing whether another organic component is a counterion or a drug.
        if len(organic) != 1:
            result["structure_flags"].append("multiple_organic_fragments" if len(organic) > 1 else "no_organic_fragment")
        parent = rdMolStandardize.FragmentParent(clean, skipStandardize=True)
        parent = UNCHARGER.uncharge(parent)
        Chem.SanitizeMol(parent)
        result["parent_smiles"] = Chem.MolToSmiles(parent, isomericSmiles=True)
        result["parent_inchikey"] = Chem.MolToInchiKey(parent)
        if not result["parent_inchikey"]:
            result["structure_flags"].append("missing_inchikey")
        result["connectivity_smiles"] = Chem.MolToSmiles(parent, isomericSmiles=False)
        result["scaffold"] = MurckoScaffold.MurckoScaffoldSmiles(mol=parent, includeChirality=False) or "ACYCLIC"
        result["heavy_atoms"] = parent.GetNumHeavyAtoms()
        result["molecular_weight"] = Descriptors.MolWt(parent)
        result["fragment_count"] = len(fragments)
        result["structure_changed"] = result["canonical_raw_smiles"] != result["parent_smiles"]
        result["structure_eligible"] = not result["structure_flags"]
        return result
    except (ValueError, RuntimeError) as exc:
        return {**result, "structure_flags": ["standardization_error"], "structure_error": str(exc), "structure_eligible": False}


def consolidate(records):
    """One standardized stereochemical parent per sample; conflicting labels excluded."""
    inchi_parents = defaultdict(set)
    for row in records:
        if row["structure_eligible"]:
            inchi_parents[row["parent_inchikey"]].add(row["parent_smiles"])
    ambiguous = {key for key, parents in inchi_parents.items() if len(parents) > 1}
    for row in records:
        if row.get("parent_inchikey") in ambiguous:
            row["structure_flags"].append("same_inchikey_different_parent_smiles")
            row["structure_eligible"] = False
    groups = defaultdict(list)
    for row in records:
        if row["structure_eligible"]:
            groups[row["parent_smiles"]].append(row)
    compounds = []
    for parent, rows in sorted(groups.items()):
        labels = {r["activity"] for r in rows}
        conflict = len(labels) != 1
        for row in rows:
            row["duplicate_parent_count"] = len(rows)
            row["conflicting_parent_labels"] = conflict
            row["eligible"] = not conflict and not any("author_label_reconciliation_issue" in r["measurement_flags"] for r in rows)
        if not all(r["eligible"] for r in rows):
            continue
        representative = rows[0]
        compounds.append({"compound_key": representative["parent_inchikey"], "parent_smiles": parent,
                          "connectivity_smiles": representative["connectivity_smiles"], "scaffold": representative["scaffold"],
                          "activity": representative["activity"], "record_ids": [r["record_id"] for r in rows],
                          "source_rows": [r["source_row"] for r in rows], "source_compound_ids": [r["source_compound_id"] for r in rows],
                          "measurement_count": len(rows), "heavy_atoms": representative["heavy_atoms"],
                          "molecular_weight": representative["molecular_weight"], "np_status": "unknown", "actinomycete_status": "unknown"})
    for row in records:
        if not row["structure_eligible"]:
            row["eligible"] = False
    return compounds
