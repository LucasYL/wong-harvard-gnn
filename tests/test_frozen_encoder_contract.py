"""Feature extraction must preserve molecule order and frozen parameters."""
import numpy as np
import pytest

torch = pytest.importorskip("torch")
pytest.importorskip("chemprop")

from chemprop import nn
from antibacterial.frozen_encoder import encode_rows, parameter_digest


def test_frozen_embeddings_are_ordered_batch_invariant_and_unchanged():
    torch.manual_seed(17)
    encoder = nn.BondMessagePassing(d_h=16, depth=3).requires_grad_(False).eval()
    aggregation = nn.MeanAggregation().eval()
    rows = [{"parent_smiles": s} for s in ("CCO", "c1ccccc1", "CC(=O)O")]
    before = parameter_digest(encoder)
    original = np.concatenate(list(encode_rows(rows, encoder, aggregation, 2)))
    reversed_one_at_a_time = np.concatenate(list(encode_rows(rows[::-1], encoder, aggregation, 1)))[::-1]
    assert original.shape == (3, 16)
    np.testing.assert_allclose(original, reversed_one_at_a_time, rtol=1e-5, atol=1e-6)
    assert parameter_digest(encoder) == before
    assert all(p.grad is None and not p.requires_grad for p in encoder.parameters())
