import json

import numpy as np
import pytest

torch = pytest.importorskip("torch")
pytest.importorskip("chemprop")

from descriptastorus.descriptors import rdNormalizedDescriptors
from antibacterial import descriptors
from antibacterial.neural import make_loader
from antibacterial.neural_descriptors import dataset, model, mean_scores, all_metrics


def test_real_descriptors_and_singleton_batch_are_preserved():
    g = rdNormalizedDescriptors.RDKit2DNormalized()
    rows = [{"parent_smiles": s, "activity": y} for s, y in [("CCO", 1), ("CCN", 0), ("CCC", 0)]]
    x = np.stack([descriptors.vector(g, r["parent_smiles"])[0] for r in rows])
    batches = list(make_loader(dataset(rows, x), 2))
    assert [len(b[0]) for b in batches] == [2, 1]
    np.testing.assert_array_equal(torch.cat([b[2] for b in batches]).numpy(), x)
    np.testing.assert_array_equal(torch.cat([b[3] for b in batches]).numpy().ravel(), [1, 0, 0])


def test_descriptor_cache_rejects_reordered_ids(tmp_path, monkeypatch):
    monkeypatch.setattr(descriptors, "CACHE", tmp_path)
    monkeypatch.setattr(descriptors, "identity", lambda: {"fixed": True})
    (tmp_path / "manifest.json").write_text(json.dumps({"complete": True, "identity": {"fixed": True},
                                                      "partitions": {"train": {"hashes": {}}}}))
    (tmp_path / "train_keys.json").write_text('["a", "b"]')
    with pytest.raises(ValueError, match="identity/order"):
        descriptors.load_features("train", [{"compound_key": "b"}, {"compound_key": "a"}])


def test_final_test_partition_is_forbidden():
    with pytest.raises(ValueError, match="only train and validation"):
        descriptors.read_rows("test")


def test_descriptors_actually_enter_prediction_and_receive_gradients():
    config = json.loads(descriptors.CONFIG.read_text())
    torch.manual_seed(17)
    network = model(config)
    rows = [{"parent_smiles": "CCO", "activity": 1}, {"parent_smiles": "CCO", "activity": 1}]
    x = np.stack([np.zeros(200), np.ones(200)]).astype(np.float32)
    bmg, vd, xd, _, *_ = next(iter(make_loader(dataset(rows, x), 2)))
    xd.requires_grad_()
    fingerprint = network.fingerprint(bmg, vd, xd)
    assert fingerprint.shape == (2, 500)
    scores = network.predictor.train_step(fingerprint)
    assert not torch.isclose(scores[0], scores[1]).all()
    scores.sum().backward()
    assert torch.isfinite(xd.grad).all() and torch.count_nonzero(xd.grad) > 0


def test_prediction_averaging_and_metric_conventions():
    scores = mean_scores([[0.8, 0.2], [0.4, 0.6], [0.6, 0.4]])
    np.testing.assert_allclose(scores, [0.6, 0.4])
    result = all_metrics([0, 0, 1, 1], [0.1, 0.4, 0.35, 0.8], list('abcd'), [2])
    assert result['average_precision'] == pytest.approx(5 / 6)
    assert result['trapezoidal_pr_auc'] == pytest.approx(19 / 24)
    with pytest.raises(ValueError):
        mean_scores([[0.1, float('nan')]])
