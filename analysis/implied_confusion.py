"""Table 3 of our report: confusion counts implied by the paper's reported metrics (n = 500),
versus the exact counts from our own predictions."""
import argparse
import os
import sys
import pathlib

sys.path.insert(0, str(pathlib.Path(__file__).resolve().parents[1] / "src"))

import pandas as pd
from common import LANGS, PAPER_TABLE10, EXPECTED_TEST_ROWS, labels_to_binary, LABEL_COL


def solve(acc, prec, rec, n=EXPECTED_TEST_ROWS, tol=0.006):
    """Find all integer confusion matrices consistent with accuracy, precision and recall
    (percent, reported to 2 decimals) on n examples. Returns a list of (P, TP, FP, TN, FN),
    where P is the number of gold-faithful pairs."""
    out = []
    for P in range(1, n):
        TP = round(rec / 100 * P)
        if TP == 0 or abs(100 * TP / P - rec) > tol:
            continue
        FP = round(TP / (prec / 100)) - TP
        if FP < 0 or abs(100 * TP / (TP + FP) - prec) > tol:
            continue
        TN = (n - P) - FP
        if TN < 0 or abs(100 * (TP + TN) / n - acc) > tol:
            continue
        out.append((P, TP, FP, TN, P - TP))
    return out


def counts_from_preds(path):
    """Exact (P, TP, FP, TN, FN) from a saved predictions CSV."""
    d = pd.read_csv(path)
    y = labels_to_binary(d[LABEL_COL])
    p = d["pred"].astype(int).values
    TP = int(((y == 1) & (p == 1)).sum()); FP = int(((y == 0) & (p == 1)).sum())
    TN = int(((y == 0) & (p == 0)).sum()); FN = int(((y == 1) & (p == 0)).sum())
    return (int(y.sum()), TP, FP, TN, FN)


def main():
    """Write results/table3_confusion.csv with paper-implied and our counts per language."""
    ap = argparse.ArgumentParser()
    ap.add_argument("--results-dir", default="results")
    a = ap.parse_args()
    rows = []
    for lang in LANGS:
        sol = solve(*PAPER_TABLE10[lang][:3])
        if sol:
            P, TP, FP, TN, FN = sol[0]
            rows.append(dict(lang=lang, source="paper (implied)", faithful=P, TP=TP, FN=FN, FP=FP, TN=TN,
                             errors=FN + FP, n_solutions=len(sol)))
        pred_path = os.path.join(a.results_dir, "table10", f"{lang}_preds.csv")
        if os.path.exists(pred_path):
            P, TP, FP, TN, FN = counts_from_preds(pred_path)
            rows.append(dict(lang=lang, source="ours (exact)", faithful=P, TP=TP, FN=FN, FP=FP, TN=TN,
                             errors=FN + FP, n_solutions=1))
    out = pd.DataFrame(rows)
    os.makedirs(a.results_dir, exist_ok=True)
    out.to_csv(os.path.join(a.results_dir, "table3_confusion.csv"), index=False)
    print(out.to_string(index=False))


if __name__ == "__main__":
    main()
