import copy

import numpy as np
import pytest
from rdkit import DataStructs

from antibacterial.baseline import checked_probability, metrics, neighbour_predict
from antibacterial.chemistry import consolidate, standardize
from antibacterial.splitting import assert_no_leakage, scaffold_split


def row(smiles, activity, row_number):
    return {"activity": activity, "record_id": str(row_number), "source_row": row_number,
            "source_compound_id": str(row_number), "measurement_flags": [], **standardize(smiles)}


def test_salt_parent_and_charged_form_agree():
    assert standardize("C[NH3+].[Cl-]")["parent_smiles"] == standardize("CN")["parent_smiles"]


def test_organic_mixture_is_flagged_not_silently_a_single_molecule():
    assert not standardize("CCO.CN")["structure_eligible"]


def test_stereochemistry_retained_but_connectivity_group_agrees():
    a, b = standardize("N[C@@H](C)C(=O)O"), standardize("N[C@H](C)C(=O)O")
    assert a["parent_smiles"] != b["parent_smiles"]
    assert a["connectivity_smiles"] == b["connectivity_smiles"]
    assert a["scaffold"] == b["scaffold"]


def test_invalid_structure_excluded():
    assert not standardize("not_a_structure")["structure_eligible"]


def test_conflicting_measurements_never_majority_voted():
    data = [row("CN", 1, 2), row("C[NH3+].[Cl-]", 0, 3), row("CN", 1, 4)]
    assert consolidate(data) == []
    assert all(r["conflicting_parent_labels"] and not r["eligible"] for r in data)


def test_consistent_duplicates_retain_all_sources():
    data = [row("CN", 1, 2), row("C[NH3+].[Cl-]", 1, 3)]
    merged = consolidate(data)
    assert len(merged) == 1
    assert merged[0]["source_rows"] == [2, 3]


def test_same_inchikey_different_structures_require_review_before_split():
    data = [row("Oc1ccccn1", 0, 2), row("O=c1cccc[nH]1", 0, 3)]
    assert data[0]["parent_inchikey"] == data[1]["parent_inchikey"]
    assert data[0]["parent_smiles"] != data[1]["parent_smiles"]
    assert consolidate(data) == []
    assert all("same_inchikey_different_parent_smiles" in r["structure_flags"] for r in data)


def sample_groups():
    return [{"scaffold": f"scaffold_{i // 3}", "parent_smiles": str(i), "connectivity_smiles": str(i),
             "compound_key": str(i), "activity": i % 2} for i in range(300)]


def test_split_independent_of_labels_and_input_order():
    a = sample_groups()
    b = list(reversed(copy.deepcopy(a)))
    for r in b:
        r["activity"] = 1 - r["activity"]
    for rows in (a, b):
        scaffold_split(rows, 42, {"train": 0.8, "validation": 0.1, "test": 0.1})
    assert {r["compound_key"]: r["split"] for r in a} == {r["compound_key"]: r["split"] for r in b}


def test_leakage_guard_catches_stereoisomer_identity_overlap():
    data = sample_groups()
    scaffold_split(data, 42, {"train": 0.8, "validation": 0.1, "test": 0.1})
    a = next(r for r in data if r["split"] == "train")
    b = next(r for r in data if r["split"] == "test")
    b["connectivity_smiles"] = a["connectivity_smiles"]
    with pytest.raises(ValueError, match="connectivity"):
        assert_no_leakage(data)


def test_top_k_ties_are_reported_without_using_labels_to_rank():
    result = metrics([0, 1, 1, 0], [0.5] * 4, ["a", "b", "c", "d"], [1, 2])
    assert result["average_precision"] == 0.5
    assert result["top_k"]["1"]["hits"] == 0
    assert result["top_k"]["1"]["tie_averaged_precision"] == 0.5
    assert result["top_k"]["2"]["tie_averaged_enrichment"] == 1


def test_neighbours_include_boundary_ties_and_zero_similarity_falls_back_to_prior():
    a, b, q = (DataStructs.CreateFromBitString(s) for s in ("1100", "1100", "0011"))
    scores, _, _, used = neighbour_predict([a, b], [a, q], [0, 1], 1)
    assert np.allclose(scores, [0.5, 0.5])
    assert used == [2, 2]


def test_float_roundoff_is_corrected_but_real_probability_errors_fail():
    assert checked_probability(1 + 2e-16) == 1
    assert checked_probability(-2e-16) == 0
    for value in (1.01, -0.01, float("nan"), float("inf")):
        with pytest.raises(ValueError, match="outside"):
            checked_probability(value)
