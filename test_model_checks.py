"""Sanity and reproduction checks that need the dataset and download Hugging Face models.
Run with:  bash scripts/run_tests.sh   (skip with --fast)."""
import numpy as np
import pytest

from common import (DOC_COL, SUM_COL, LABEL_COL, THRESHOLD, PAPER_TABLE10, labels_to_binary,
                    load_test, load_model, score_pairs, metrics, test_file)

pytestmark = pytest.mark.model


def _need_data(lang):
    """Skip (not fail) when the dataset has not been downloaded yet."""
    try:
        test_file(lang)
    except FileNotFoundError:
        pytest.skip("dataset missing: run scripts/download_data.sh")


def test_shuffled_documents_reduce_faithful_rate():
    """Sanity check: pairing summaries with unrelated documents must lower the share predicted faithful."""
    _need_data("es_XX")
    d = load_test("es_XX", n=100)
    docs, sums = d[DOC_COL].astype(str).values, d[SUM_COL].astype(str).values
    tok, model, device = load_model("es_XX")
    original = (score_pairs(tok, model, docs, sums, device) > THRESHOLD).mean()
    shuffled = (score_pairs(tok, model, np.roll(docs, 37), sums, device) > THRESHOLD).mean()
    assert shuffled < original


def test_spanish_f1_matches_paper_within_one_point():
    """Reproduction check: full Spanish test set F1 is within 1 point of the paper's 96.96."""
    _need_data("es_XX")
    d = load_test("es_XX")
    y = labels_to_binary(d[LABEL_COL])
    tok, model, device = load_model("es_XX")
    pred = (score_pairs(tok, model, d[DOC_COL].astype(str), d[SUM_COL].astype(str), device) > THRESHOLD).astype(int)
    assert abs(metrics(y, pred)["f1"] - PAPER_TABLE10["es_XX"][3]) <= 1.0
