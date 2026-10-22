"""Run in the neural environment; skipped in the lean baseline environment."""
import numpy as np
import pytest

torch = pytest.importorskip("torch")
pytest.importorskip("chemprop")

from sklearn.metrics import average_precision_score
from antibacterial.neural import make_dataset, make_loader
from antibacterial.neural_metrics import AveragePrecision


def test_epoch_ap_matches_sklearn_with_ties_across_batches():
    labels = torch.tensor([[1], [0], [1], [0], [0], [1]], dtype=torch.float32)
    scores = torch.tensor([[0.8], [0.8], [0.6], [0.2], [0.2], [0.2]])
    metric = AveragePrecision()
    for a, b in ((0, 2), (2, 6)):
        metric.update(scores[a:b], labels[a:b], torch.ones_like(labels[a:b], dtype=torch.bool))
    assert float(metric.compute()) == pytest.approx(average_precision_score(labels.numpy(), scores.numpy()), abs=1e-7)


def test_prediction_loader_keeps_singleton_and_input_order():
    rows = [{"parent_smiles": s, "activity": y} for s, y in (("C", 1), ("CC", 0), ("CCC", 1))]
    batches = list(make_loader(make_dataset(rows), batch_size=2))
    assert [len(batch[0]) for batch in batches] == [2, 1]
    np.testing.assert_array_equal(torch.cat([batch[3] for batch in batches]).numpy().ravel(), [1, 0, 1])
