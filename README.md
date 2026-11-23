# AI504 Assignment 2: Reproducing mFACT (Qiu et al., EMNLP 2023)

Group [09] | [Ismail Zaid Alkilani], [Alfarouq Alsharif], [Lakshmi Vijay]
---

## 1. Paper reproduced and claim tested

**Paper.** Yifu Qiu, Yftah Ziser, Anna Korhonen, Edoardo M. Ponti, and Shay B. Cohen (2023). *Detecting and Mitigating Hallucinations in Multilingual Summarisation.* Proceedings of EMNLP 2023, pages 8914-8932.
- Paper: https://aclanthology.org/2023.emnlp-main.551/
- Original code, data and models: https://github.com/yfqiu-nlp/mfact-summ

**Claim tested.** Multilingual BERT classifiers fine-tuned on translated pseudo-labels (mFACT) reach a mean F1 of 95.2 (SD 1.8) and mean accuracy of 95.3 across Chinese, Spanish, French, Hindi, Vietnamese and Turkish on the translated test sets (500 pairs per language). This is Table 1 of the paper, with per-language results in its Table 10.

**Scope.** We evaluate the authors' released checkpoints on the released test sets. We do **not** retrain the classifiers and do **not** reproduce the summarisation experiments (the paper's Table 4) or the human evaluation.

**Our headline result.** Mean F1 95.65 (SD 1.18) against 95.2 in the paper. Five of six languages match the paper to two decimals; Chinese differs (our F1 95.10 against 92.53). Verdict: claim supported for the released artefacts. See the report for details.

**Extra analyses (ours).** (a) Confusion matrices implied by the paper's metrics; (b) a document-dependence probe (shuffled and empty documents); (c) a decision-threshold sweep.

---

## 2. Environment

| Item | Value |
|---|---|
| Python | 3.13 |
| Dependencies | `requirements.txt` (pinned versions)|
| Hardware used for reported results | Google Colab, T4 |
| Seeds | Inference is deterministic. `set_seed(0)` in `src/common.py` fixes Python, NumPy and PyTorch seeds anyway |

Setup (Linux/macOS, or Git Bash / WSL on Windows, since the scripts are bash):

```bash
python -m pip install -r requirements.txt
```

---

## 3. Data

| Item | Details |
|---|---|
| Dataset | Authors' translated faithfulness dataset (`{lang}-translated-{train,valid,test}.csv`) |
| Download | https://drive.google.com/drive/folders/1WbTqPWLGTGkJFgVxVehPMjM0rYGUwwG7 |
| Models | `yfqiu-nlp/mFACT-{zh_CN,es_XX,fr_XX,hi_IN,vi_VN,tr_TR}` on Hugging Face (MIT licence as shown on the model pages) |
| Size | Test files: 500 pairs per language. Models: about 700 MB each |

**Download and preprocessing (one command):**

```bash
bash scripts/download_data.sh
```

This script (1) downloads the Drive folder with `gdown` into `dataset/xsum-translation/`, (2) checks that each language's test file has 500 rows, and (3) checks the columns `premise`, `hypothesis`, `label`.

**Preprocessing applied (nothing else is done to the data):**
- Language prefixes map to model IDs: `zh`->`zh_CN`, `es`->`es_XX`, `fr`->`fr_XX`, `hi`->`hi_IN`, `vi`->`vi_VN`, `tr`->`tr_TR`.
- `premise` is the document and `hypothesis` is the summary. Labels: `entailment` = faithful (1), `contradiction` = hallucinated (0). This mapping is not stated in the original README; it is our inference, supported by the exact agreement with the paper in five languages.
- Tokenisation: document and summary encoded as a sentence pair, `truncation="only_first"`, `max_length=512`.

Data and model weights are **not committed**. They are downloaded by the script (see `.gitignore`).

---

## 4. Commands: one per reported result

Run from the repository root. Each command writes to `results/`.

| Report item | Command | Output |
|---|---|---|
| Tables 1 and 2 (per-language and summary metrics vs paper) | `bash scripts/reproduce_tables_1_2.sh` | `results/table10/*_preds.csv`, `table10_comparison.csv`, `table1_comparison.csv` |
| Table 3 (implied confusion matrices) | `bash scripts/reproduce_table3_confusion.sh` | `results/table3_confusion.csv` |
| Table 4 (document-dependence probe) | `bash scripts/reproduce_table4_probe.sh` | `results/probe_document_use.csv` |
| Table 5 (threshold sweep) | `bash scripts/reproduce_table5_sweep.sh` | `results/threshold_sweep_f1.csv` |
| Everything | `bash scripts/run_all.sh` | all of the above |

The scripts call Python in `src/` and `analysis/`, ported from the group's Colab notebooks. Table 3 recovers the paper's confusion counts from its accuracy, precision and recall at n = 500 and compares them with exact counts from our predictions. The Table 3 and Table 5 scripts read `results/table10/` written by the first command, and `run_all.sh` runs everything in order. Per-language timings are written to `results/table10/timing.csv`. Optional: `bash scripts/reproduce_table4_probe.sh --random-shuffles 5` adds five random-offset shuffles so variance can be reported.


```bash
bash scripts/reproduce_tables_1_2.sh --quick    # Spanish and Hindi only, 100 randomly sampled pairs each (fixed seed); writes results/quick/
```


---

## 5. Expected output 

Reproduction tolerance: within 1 percentage point of the values below (rubric guideline for "reproduced").

| Command | Expected result |
|---|---|
| `reproduce_tables_1_2.sh` | F1 (ours): zh 95.10, es 96.96, fr 95.58, hi 93.88, vi 95.41, tr 96.97. Accuracy: 95.2, 97.0, 95.6, 94.0, 95.4, 97.0. Mean (SD): acc 95.70 (1.15), precision 95.95 (1.24), recall 95.36 (1.63), F1 95.65 (1.18). Paper: F1 mean 95.2 (1.8); five languages identical to the paper, zh differs |
| `reproduce_table3_confusion.sh` | Each language has 248 faithful and 252 hallucinated pairs. Errors (ours): zh 24, es 15, fr 22, hi 30, vi 23, tr 15. Paper's zh row implies 37 errors | 
| `reproduce_table4_probe.sh` | Accuracy, original / shuffled document / empty document: zh 95.2/54.2/53.0, es 97.0/52.2/50.4, fr 95.6/52.6/50.4, hi 94.0/57.8/50.4, vi 95.4/54.6/50.4, tr 97.0/53.4/50.4. Mean 95.70 / 54.13 / 50.83 |
| `reproduce_table5_sweep.sh` | Mean F1 over languages: 95.65 at threshold 0.5, 95.90 at 0.7. F1 varies by at most 0.96 points between thresholds 0.3 and 0.7 |

The first run downloads each model from Hugging Face. Later runs use the local cache.

---

## 6. Code reused from the authors vs written by the group

| Component | Origin |
|---|---|
| Translated faithfulness dataset | **Authors** (Google Drive) |
| Six trained classifiers (`yfqiu-nlp/mFACT-*`) | **Authors** (Hugging Face) |
| Method and published numbers (Tables 1, 10) | **Authors** (paper, repo) |
| Evaluation loop: load checkpoint, build input, batch scoring, 0.5 threshold | **Group** (drafted with AI assistance, adapted and run by the group) |
| Metrics (scikit-learn), comparison tables, mean and SD calculation | **Group** |
| Document-dependence probe and threshold sweep | **Group** |



---

## 7. Tests

```bash
bash scripts/run_tests.sh --fast   # unit tests only; no data or models needed
bash scripts/run_tests.sh          # all tests; needs the dataset and downloads models
```

| Test file | What it checks | Needs data? |
|---|---|---|
| `tests/test_metrics_and_labels.py` (4 tests) | Metrics on toy examples; `entailment` -> 1 and `contradiction` -> 0; unknown labels raise an error | No |
| `tests/test_implied_confusion.py` (2 tests) | Confusion solver reproduces the hand-checked Spanish and Chinese rows (Chinese paper row implies 37 errors) | No |
| `tests/test_model_checks.py` (2 tests, marker `model`) | (a) Sanity: shuffled documents lower the share predicted faithful; (b) Spanish F1 within 1 point of the paper's 96.96 | Yes |


---

## Findings and advice for future reproducers

- **What worked.** The released checkpoints and test files reproduce the paper exactly in five of six languages with a short custom loop.
- **Chinese differs.** The released Chinese checkpoint makes 24 errors where the paper's row implies 37. Possible causes (untested): a different checkpoint or test file. Ask the authors or compare checkpoint history.
- **Undocumented conventions.** The label strings (`entailment` = faithful), column names (`premise`, `hypothesis`, `label`) and language-code mapping are not in the original README.
- **Original README problems (found by reading, not by running).** `pip install requirements.txt` lacks `-r`; the training command leaves `${pretrained}` undefined; the commands use bash-only syntax; the inference example links to a different GitHub account.
- **Advice.** Confirm the label mapping first on a single language. Reproduce Table 10 before anything else. Retrain over several seeds only if you have GPU time.

## Repository layout

```
README.md                  this file
requirements.txt           dependencies 
pytest.ini                 registers the 'model' test marker
src/                       common.py, evaluate_classifiers.py, check_data.py
analysis/                  implied_confusion.py, probe_document_use.py, threshold_sweep.py
scripts/                   one script per reported result, run_all.sh, run_tests.sh
tests/                     unit and sanity tests
results/                   outputs and logs (one file per reported table)
```

## Acknowledgements

All models, data and the method are by Qiu et al. (2023); see the citation above. 
