"""Table 4 of our report: does the classifier depend on the document?

Conditions: original document; shuffled document (np.roll by a fixed offset, so every pair is
mismatched); empty document. Optionally K extra shuffles with random offsets to report variance.
"""
import argparse
import os
import sys
import pathlib

sys.path.insert(0, str(pathlib.Path(__file__).resolve().parents[1] / "src"))

import numpy as np
import pandas as pd
from common import (LANGS, DATA_DIR, DOC_COL, SUM_COL, LABEL_COL, THRESHOLD, labels_to_binary,
                    load_test, load_model, score_pairs, set_seed)


def probe_rows(lang, conds, sums, y, tok, model, device):
    """Score each condition and return one summary row per condition."""
    rows = []
    for name, docs in conds.items():
        pred = (score_pairs(tok, model, docs, sums, device) > THRESHOLD).astype(int)
        rows.append(dict(lang=lang, condition=name, acc=100 * (pred == y).mean(),
                         pct_pred_faithful=100 * pred.mean(),
                         on_gold_faithful=100 * pred[y == 1].mean(),
                         on_gold_halluc=100 * pred[y == 0].mean()))
    return rows


def main():
    """Run the probe for all six languages and write results/probe_document_use.csv."""
    ap = argparse.ArgumentParser()
    ap.add_argument("--data-dir", default=DATA_DIR)
    ap.add_argument("--results-dir", default="results")
    ap.add_argument("--offset", type=int, default=137, help="fixed shuffle offset used in the report")
    ap.add_argument("--random-shuffles", type=int, default=0, help="extra shuffles with random offsets")
    ap.add_argument("--seed", type=int, default=0)
    a = ap.parse_args()
    set_seed(a.seed)
    rng = np.random.default_rng(a.seed)

    rows = []
    for lang in LANGS:
        d = load_test(lang, a.data_dir)
        docs = d[DOC_COL].astype(str).values
        sums = d[SUM_COL].astype(str).values
        y = labels_to_binary(d[LABEL_COL])
        conds = {"original": docs, "shuffled_docs": np.roll(docs, a.offset),
                 "summary_only": np.array([""] * len(docs))}
        for k in range(a.random_shuffles):       # offsets in 1..n-1 never leave a pair matched
            conds[f"shuffled_random_{k + 1}"] = np.roll(docs, int(rng.integers(1, len(docs))))
        tok, model, device = load_model(lang)
        rows += probe_rows(lang, conds, sums, y, tok, model, device)
        del model
    out = pd.DataFrame(rows).round(1)
    os.makedirs(a.results_dir, exist_ok=True)
    out.to_csv(os.path.join(a.results_dir, "probe_document_use.csv"), index=False)
    print(out.pivot_table(index="lang", columns="condition", values="acc"))


if __name__ == "__main__":
    main()
