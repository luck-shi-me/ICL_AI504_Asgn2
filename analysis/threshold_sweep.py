"""Table 5 of our report: F1 as a function of the decision threshold, from stored scores
(no model is rerun). Note: the 'best' threshold is chosen on the test set, so it is optimistic."""
import argparse
import os
import sys
import pathlib

sys.path.insert(0, str(pathlib.Path(__file__).resolve().parents[1] / "src"))

import numpy as np
import pandas as pd
from sklearn.metrics import f1_score
from common import LANGS, LABEL_COL, labels_to_binary

THRESHOLDS = np.round(np.arange(0.1, 0.95, 0.1), 1)


def sweep_language(d):
    """Return F1 (percent) at each threshold for one language's prediction file."""
    y = labels_to_binary(d[LABEL_COL])
    return [100 * f1_score(y, (d["p_faithful"] > t).astype(int)) for t in THRESHOLDS]


def summarise(sweep):
    """Per-language sensitivity statistics (the columns of Table 5)."""
    rows = []
    for lang in sweep.columns:
        s = sweep[lang]
        mid = s.loc[0.3:0.7]
        rows.append(dict(lang=lang, f1_at_0_5=s.loc[0.5], range_0_3_to_0_7=mid.max() - mid.min(),
                         range_0_1_to_0_9=s.max() - s.min(), best_threshold=s.idxmax(),
                         f1_at_best=s.max(), gain_over_0_5=s.max() - s.loc[0.5]))
    return pd.DataFrame(rows).round(2)


def main():
    """Read results/table10/*_preds.csv and write the sweep and its summary."""
    ap = argparse.ArgumentParser()
    ap.add_argument("--results-dir", default="results")
    a = ap.parse_args()
    sweep = pd.DataFrame({lang: sweep_language(pd.read_csv(os.path.join(a.results_dir, "table10", f"{lang}_preds.csv")))
                          for lang in LANGS}, index=THRESHOLDS).round(2)
    sweep.to_csv(os.path.join(a.results_dir, "threshold_sweep_f1.csv"))
    summ = summarise(sweep)
    summ.to_csv(os.path.join(a.results_dir, "threshold_sweep_summary.csv"), index=False)
    print(sweep); print(summ.to_string(index=False))
    print("Mean F1 over languages by threshold:", sweep.mean(axis=1).round(2).to_dict())


if __name__ == "__main__":
    main()
