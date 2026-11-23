"""Tables 1 and 2 of our report: run the released mFACT classifiers on the translated test sets
and compare accuracy/precision/recall/F1 with the paper's Table 10 and Table 1.

Usage:
    python src/evaluate_classifiers.py            # full run, writes results/table10/
    python src/evaluate_classifiers.py --quick    # reduced run: 2 languages, 100 sampled pairs
"""
import argparse
import os
import time

import pandas as pd

from common import (LANGS, DATA_DIR, DOC_COL, SUM_COL, LABEL_COL, THRESHOLD, BATCH, METRICS,
                    PAPER_TABLE10, PAPER_TABLE1, labels_to_binary, load_test, load_model,
                    score_pairs, metrics, set_seed)


def evaluate_language(lang, data_dir, out_dir, n=None, bs=BATCH, device=None):
    """Score one language's test set, save per-example predictions, and return its metrics."""
    d = load_test(lang, data_dir, n=n)
    y = labels_to_binary(d[LABEL_COL])
    tok, model, device = load_model(lang, device)
    d["p_faithful"] = score_pairs(tok, model, d[DOC_COL].astype(str), d[SUM_COL].astype(str), device, bs=bs)
    d["pred"] = (d["p_faithful"] > THRESHOLD).astype(int)
    d.to_csv(os.path.join(out_dir, f"{lang}_preds.csv"), index=False)
    return dict(lang=lang, n=len(d), **metrics(y, d["pred"].values))


def main():
    """Run all (or the quick subset of) languages and write the comparison tables."""
    ap = argparse.ArgumentParser()
    ap.add_argument("--quick", action="store_true", help="2 languages, 100 sampled pairs, writes results/quick/")
    ap.add_argument("--data-dir", default=DATA_DIR)
    ap.add_argument("--out-dir", default=None)
    ap.add_argument("--batch-size", type=int, default=BATCH)
    ap.add_argument("--device", default=None)
    a = ap.parse_args()
    set_seed(0)

    langs = ["es_XX", "hi_IN"] if a.quick else list(LANGS)
    n = 100 if a.quick else None
    out_dir = a.out_dir or ("results/quick" if a.quick else "results/table10")
    os.makedirs(out_dir, exist_ok=True)

    rows, timing = [], []
    for lang in langs:
        t0 = time.perf_counter()
        r = evaluate_language(lang, a.data_dir, out_dir, n=n, bs=a.batch_size, device=a.device)
        timing.append(dict(lang=lang, seconds=round(time.perf_counter() - t0, 1)))
        rows.append(r)
        print({k: (round(v, 2) if isinstance(v, float) else v) for k, v in r.items()})

    res = pd.DataFrame(rows).set_index("lang")
    for i, m in enumerate(METRICS):
        res[m + "_paper"] = [PAPER_TABLE10[l][i] for l in res.index]
        res[m + "_delta"] = res[m] - res[m + "_paper"]
    res.round(2).to_csv(os.path.join(out_dir, "table10_comparison.csv"))
    pd.DataFrame(timing).to_csv(os.path.join(out_dir, "timing.csv"), index=False)

    if a.quick:
        print("Quick run: a 100-pair sample is NOT comparable with the paper; this only checks the pipeline.")
        return
    t1 = pd.DataFrame({"mean": res[METRICS].mean(), "sd": res[METRICS].std()})   # sample SD (n-1)
    t1["paper_mean"] = [PAPER_TABLE1[m][0] for m in METRICS]
    t1["paper_sd"] = [PAPER_TABLE1[m][1] for m in METRICS]
    t1.round(2).to_csv(os.path.join(out_dir, "table1_comparison.csv"))
    print(t1.round(2))


if __name__ == "__main__":
    main()
