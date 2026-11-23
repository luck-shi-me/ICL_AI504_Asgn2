"""Unit tests for the metric and label-mapping helpers (no data or models needed)."""
import pytest

from common import labels_to_binary, metrics


def test_metrics_on_toy_example():
    """2 of 4 correct; faithful is the positive class, so precision, recall and F1 are all 50."""
    m = metrics([1, 1, 0, 0], [1, 0, 0, 1])
    assert m["acc"] == pytest.approx(50.0)
    assert m["prec"] == pytest.approx(50.0)
    assert m["rec"] == pytest.approx(50.0)
    assert m["f1"] == pytest.approx(50.0)


def test_metrics_perfect_prediction():
    """Perfect predictions give 100 on every metric."""
    m = metrics([1, 0, 1, 0], [1, 0, 1, 0])
    assert all(m[k] == pytest.approx(100.0) for k in ("acc", "prec", "rec", "f1"))


def test_label_mapping():
    """entailment -> 1 (faithful), contradiction -> 0 (hallucinated)."""
    assert list(labels_to_binary(["entailment", "contradiction", "entailment"])) == [1, 0, 1]


def test_unknown_label_fails_loudly():
    """An unexpected label string must raise instead of being silently treated as 0."""
    with pytest.raises(ValueError):
        labels_to_binary(["entailment", "neutral"])
