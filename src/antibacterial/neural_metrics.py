"""Chemprop-compatible non-interpolated AP, matching the baseline metric."""
from chemprop.nn.metrics import ClassificationMixin
from torchmetrics.classification import BinaryAveragePrecision


class AveragePrecision(ClassificationMixin, BinaryAveragePrecision):
    alias = "ap"
    higher_is_better = True
