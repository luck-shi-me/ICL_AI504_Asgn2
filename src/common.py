"""Shared helpers for the mFACT reproduction (group-written code).

This module re-implements the evaluation of the authors' released mFACT classifiers.
It is NOT the authors' script (run_hallu_classification.py); see README section 6.
Heavy imports (torch, transformers) are done lazily so the pure-Python parts can be
unit-tested without them.
"""
import glob
import random

import numpy as np
import pandas as pd
from sklearn.metrics import precision_recall_fscore_support

# Language code -> prefix used in the dataset file names (e.g. es-translated-test.csv).
LANGS = {"zh_CN": "zh", "es_XX": "es", "fr_XX": "fr", "hi_IN": "hi", "vi_VN": "vi", "tr_TR": "tr"}
DATA_DIR = "dataset/xsum-translation"
MAX_LEN = 512          # paper, Appendix A.2 (maximum input length)
BATCH = 16             # our choice; affects speed and memory only
THRESHOLD = 0.5        # our choice: predict "faithful" if P(class 1) > 0.5
DOC_COL, SUM_COL, LABEL_COL = "premise", "hypothesis", "label"
POS_LABEL, NEG_LABEL = "entailment", "contradiction"   # entailment = faithful (our inference)
EXPECTED_TEST_ROWS = 500

# Paper values copied from Qiu et al. (2023). Table 10, mFACT rows: (acc, prec, rec, F1).
PAPER_TABLE10 = {
    "zh_CN": (92.60, 92.71, 92.34, 92.53), "es_XX": (97.00, 97.55, 96.37, 96.96),
    "fr_XX": (95.60, 95.20, 95.97, 95.58), "hi_IN": (94.00, 95.04, 92.74, 93.88),
    "vi_VN": (95.40, 94.47, 96.37, 95.41), "tr_TR": (97.00, 97.17, 96.77, 96.97),
}
# Paper Table 1, mFACT row: (mean, sd) over the six languages.
PAPER_TABLE1 = {"acc": (95.3, 1.7), "prec": (95.4, 1.8), "rec": (95.1, 2.0), "f1": (95.2, 1.8)}
METRICS = ["acc", "prec", "rec", "f1"]


def set_seed(seed=0):
    """Fix the Python, NumPy and (if installed) PyTorch random seeds."""
    random.seed(seed)
    np.random.seed(seed)
    try:
        import torch
        torch.manual_seed(seed)
    except ImportError:
        pass


def find_test_file(lang, data_dir=DATA_DIR):
    """Return the path of the test CSV for a language code such as 'es_XX'."""
    prefix = LANGS.get(lang, lang.split("_")[0])
    hits = glob.glob(f"{data_dir}/**/{prefix}-translated*test*.csv", recursive=True)
    if not hits:
        raise FileNotFoundError(f"No test file for {lang} under {data_dir}. Run scripts/download_data.sh")
    return hits[0]


def labels_to_binary(labels):
    """Map 'entailment' -> 1 (faithful) and 'contradiction' -> 0 (hallucinated); fail on anything else."""
    s = pd.Series(labels).astype(str).str.strip()
    bad = set(s.unique()) - {POS_LABEL, NEG_LABEL}
    if bad:
        raise ValueError(f"Unexpected label values: {sorted(bad)}")
    return (s == POS_LABEL).astype(int).values


def load_test(lang, data_dir=DATA_DIR, n=None, seed=0):
    """Load a language's test CSV; if n is given, return a reproducible random sample of n rows."""
    d = pd.read_csv(find_test_file(lang, data_dir))
    missing = {DOC_COL, SUM_COL, LABEL_COL} - set(d.columns)
    if missing:
        raise ValueError(f"{lang}: missing columns {sorted(missing)}; found {d.columns.tolist()}")
    if n is not None and n < len(d):
        d = d.sample(n=n, random_state=seed).reset_index(drop=True)
    return d


def pick_device(device=None):
    """Return 'cuda' if available, else 'cpu', unless a device is given explicitly."""
    if device:
        return device
    import torch
    return "cuda" if torch.cuda.is_available() else "cpu"


def load_model(lang, device=None):
    """Download/load the released classifier yfqiu-nlp/mFACT-<lang>; returns (tokenizer, model, device)."""
    from transformers import AutoTokenizer, AutoModelForSequenceClassification
    device = pick_device(device)
    model_id = f"yfqiu-nlp/mFACT-{lang}"
    tok = AutoTokenizer.from_pretrained(model_id)
    model = AutoModelForSequenceClassification.from_pretrained(model_id).to(device).eval()
    return tok, model, device


def score_pairs(tok, model, docs, sums, device, max_len=MAX_LEN, bs=BATCH):
    """Return P(faithful) for each (document, summary) pair; the document is truncated if too long."""
    import torch
    docs, sums = list(docs), list(sums)
    out = []
    for i in range(0, len(docs), bs):
        enc = tok(docs[i:i + bs], sums[i:i + bs], truncation="only_first", max_length=max_len,
                  padding=True, return_tensors="pt").to(device)
        with torch.no_grad():
            out += model(**enc).logits.softmax(-1)[:, 1].cpu().tolist()
    return np.array(out)


def metrics(y_true, pred):
    """Accuracy, precision, recall and F1 in percent, with 'faithful' (1) as the positive class."""
    y_true, pred = np.asarray(y_true), np.asarray(pred)
    p, r, f1, _ = precision_recall_fscore_support(y_true, pred, average="binary", pos_label=1, zero_division=0)
    return dict(acc=100 * float((y_true == pred).mean()), prec=100 * float(p), rec=100 * float(r), f1=100 * float(f1))
